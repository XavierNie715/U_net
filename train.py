import argparse
import logging
import sys
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from unet import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
os.environ["WANDB_MODE"] = "offline"

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
data_dir = './data/220mm,./data/245mm,./data/275mm_1,./data/275mm_2'


# dir_checkpoint = Path('./checkpoints/')


torch.manual_seed(42)


def train_net(net,
              device,
              dir_checkpoint: str = './checkpoints/',
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False, ):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset = BasicDataset(data_dir)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    # train_set, val_set = random_split(dataset, [n_train, n_val])

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # loss = criterion(masks_pred, true_masks) \ # 原U_Net多了一项dice_loss
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % (n_train // (10 * batch_size)) == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        # print(f'gradient: {value.grad.data.cpu()},'
                        print(f'\nMaxGradient: {value.grad.data.cpu().max()}'
                              f'\nMinGradient: {value.grad.data.cpu().min()}')
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    # val_score = evaluate(net, val_loader, device)
                    # scheduler.step(val_score)
                    net.eval()
                    num_val_batches = len(val_loader)
                    # iterate over the validation set
                    for batch in tqdm(val_loader, total=num_val_batches, desc='validation round', unit='batch',
                                      leave=False):
                        batch_data, mask_true = batch['image'], batch['mask']
                        # move images and labels to correct device and type
                        batch_data = batch_data.to(device=device, dtype=torch.float32)
                        mask_true = mask_true.to(device=device, dtype=torch.float32)

                        with torch.no_grad():
                            # predict the mask
                            mask_pred = net(batch_data)
                    net.train()
                    val_MSE = criterion(mask_pred, mask_true)

                    logging.info('Validation MSE: {}'.format(val_MSE))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation MSE': val_MSE,
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

        if save_checkpoint:
            dir = './checkpoints/' + dir_checkpoint
            torch.save(net.state_dict(), dir + 'checkpoint_epoch{}.pth'.format(epoch + 1))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--save_dir', '-sd', type=str, default=False, help='Weights saving path (in ./checkpoints/)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=2, n_classes=1, bilinear=False)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  dir_checkpoint=args.save_dir,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
