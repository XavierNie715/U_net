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
import numpy as np
from matplotlib import pyplot as plt
from utils.data_loading import BasicDataset
from utils.utils import threshold_mask, RelativeL2Error, temp_recover, std_GS, NormMSELoss
from unet import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
os.environ["WANDB_MODE"] = "offline"

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
data_list = './data/220mm', './data/245mm', './data/275mm_1', './data/275mm_2'
# data_list = './data/234mm_RE15_YF25', './data/239mm_RE10_YF35', './data/239mm_RE15_YF35', \
#             './data/260mm_RE15_YF25', './data/270mm_RE10_YF35', './data/270mm_RE15_YF35', \
#             './data/220mm', './data/245mm', './data/275mm_1', './data/275mm_2'
torch.manual_seed(42)


def train_net(net,
              device,
              dir_checkpoint: str = './checkpoints/',
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              std: bool = True,
              amp: bool = False,
              start_epoch: int = 0):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset = BasicDataset(data_dir, std)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', name=dir_checkpoint)
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, std=std,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Start epoch:     {start_epoch}
        Batch size:      {batch_size}
        Standardization: {std}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
    criterion_NormMSE = NormMSELoss()
    # criterion = nn.SmoothL1Loss()
    L2_criterion = RelativeL2Error()
    global_step = 0
    global_RMSE_error = {}

    # 5. Begin training
    for epoch in range(start_epoch, epochs):
        torch.cuda.empty_cache()
        net.train()
        epoch_loss = 0
        epoch_L2_error = 0
        epoch_RMSE_error = 0
        # epoch_L2_mask_error = 0
        epoch_MSE_error = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_Ts = batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_Ts = true_Ts.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    Ts_pred = net(images)
                    loss = criterion_MSE(Ts_pred, true_Ts)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item() / len(train_loader)
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch + 1
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        # if global_step % (n_train // (10 * batch_size)) == 0:
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            # print(f'gradient: {value.grad.data.cpu()},'
            # print(f'\nMaxGradient: {value.grad.data.cpu().max()}'
            #       f'\nMinGradient: {value.grad.data.cpu().min()}')
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        net.eval()
        num_val_batches = len(val_loader)
        # iterate over the validation set
        for batch in tqdm(val_loader, total=num_val_batches, desc='validation round', unit='batch',
                          leave=False):
            batch_data, T_true, T_ori = batch['image'], batch['mask'], batch['real_temp']

            # # calculate mask for eval
            # # Todo: 这里计算mask只考虑的batch=1的情况!
            # OH = batch_data[:, 0, :, :]
            # SVF = batch_data[:, 1, :, :]
            # mask = threshold_mask(OH.cpu().numpy()) + threshold_mask(SVF.cpu().numpy())
            # mask[mask > 0] = 1

            # move images and labels to correct device and type
            batch_data = batch_data.to(device=device, dtype=torch.float32)
            # T_true = T_true.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                # predict the T
                T_pred = net(batch_data)
                recover = True
                if recover:
                    # T_pred = temp_recover(T_ori, T_pred).to(device=device, dtype=torch.float32)
                    for x in range(batch_size):
                        T_pred[x] = temp_recover(T_ori[x], T_pred[x])
                        T_pred = T_pred.to(device=device, dtype=torch.float32)

                MSE_error = criterion_MSE(T_pred, T_ori)
                L2_error_temp = L2_criterion(T_pred, T_ori, reduct='none')
                L2_error = L2_error_temp.mean()
                # L2_mask_error = (L2_error_temp * torch.tensor(mask).to(device=device)).mean()
            epoch_L2_error += L2_error.item() / num_val_batches  # average single error for each epoch
            epoch_RMSE_error += MSE_error.sqrt().item() / num_val_batches
            # epoch_L2_mask_error += L2_mask_error.item() / num_val_batches
            epoch_MSE_error += MSE_error.item() / num_val_batches

        net.train()

        global_RMSE_error[epoch + 1] = epoch_RMSE_error
        logging.info('Rel_L2 Error: {}'.format(epoch_L2_error))
        # logging.info('masked Rel_L2 Error: {}'.format(epoch_L2_mask_error))
        logging.info('MSE Error: {}'.format(epoch_MSE_error))
        logging.info('RMSE Error: {}'.format(epoch_RMSE_error))
        logging.info('Epoch Loss: {}'.format(epoch_loss))
        logging.info('Current Minimum RMSE Error: {}, in epoch {}'.format(min(global_RMSE_error.values()),
                                                                          min(global_RMSE_error,
                                                                              key=global_RMSE_error.get)))
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation L2': epoch_L2_error,
            # 'validation masked L2': epoch_L2_mask_error,
            'validation MSE': epoch_MSE_error,
            'validation RMSE': epoch_RMSE_error,
            # 'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(T_ori[0].float().cpu()),
                'pred': wandb.Image(T_pred[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch + 1,
            **histograms
        })

        if save_checkpoint:
            dir = './checkpoints/' + dir_checkpoint
            torch.save(net.state_dict(), dir + '/checkpoint_epoch{}.pth'.format(epoch + 1))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--std', '-s', type=bool, default=True, help='Whether standardize the data')
    parser.add_argument('--std', '-s', action='store_true', help='Whether standardize the data')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--save_dir', '-sd', type=str, default=False,
                        help='Weights saving path (in ./checkpoints/), also wandb run name')
    parser.add_argument('--data_set', '-ds', type=str, default=None, nargs='+',
                        help='select train dataset(which flame height), format like 1 2 3 4')
    # parser.add_argument("--name", type=str, help="The wandb run name", default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # dataset_split = args.data_set.split(',')
    data_dir = []
    data_dir.extend([(data_list[int(item) - 1]) for item in args.data_set])
    data_dir = ','.join(data_dir)

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

    start_epoch = 0
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        start_epoch = int(args.load.split('_')[-1].split('.')[0][5:])
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  dir_checkpoint=args.save_dir,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  std=args.std,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  start_epoch=start_epoch,
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
