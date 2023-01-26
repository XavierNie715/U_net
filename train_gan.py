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
from unet.GAN_parts import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
# print(torch.cuda.is_available(), torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_name(i))
os.environ["WANDB_MODE"] = "offline"

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
data_list = './data/220mm', './data/245mm', './data/275mm_1', './data/275mm_2', \
            './data/239mm_RE10_YF35', './data/270mm_RE10_YF35', './data/239mm_RE15_YF35', './data/270mm_RE15_YF35'

# data_list = './data/234mm_RE15_YF25', './data/239mm_RE10_YF35', './data/239mm_RE15_YF35', \
#             './data/260mm_RE15_YF25', './data/270mm_RE10_YF35', './data/270mm_RE15_YF35', \
#             './data/220mm', './data/245mm', './data/275mm_1', './data/275mm_2'
torch.manual_seed(42)


def train_net(discriminator,
              generator,
              device,
              dir_checkpoint: str = './checkpoints/',
              epochs: int = 5,
              batch_size: int = 1,
              # learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              std: bool = True,
              start_epoch: int = 0):
    model = Pix2PixModel(generator_net=generator, discriminator_net=discriminator, device=device, is_train=True, )
    # 1. Create dataset
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
    experiment = wandb.init(project='GAN', resume='allow', anonymous='must', name=dir_checkpoint)
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, std=std, ))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Start epoch:     {start_epoch}
        Batch size:      {batch_size}
        Standardization: {std}
        L1 lambda:       {100}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    criterion_MSE = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    L2_criterion = RelativeL2Error()
    global_step = 0
    global_RMSE_error = {}

    # 5. Begin training
    for epoch in range(start_epoch, epochs):
        torch.cuda.empty_cache()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='snapshot') as pbar:
            for batch in train_loader:
                model.set_input(batch)
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                global_step += 1
                losses = model.get_current_losses()

                loss_message = {}
                for k, v in losses.items():
                    loss_message.update({k: v})
                    epoch_loss += v / len(train_loader)
                message = {'epoch': epoch + 1,
                           'step': global_step}
                message.update(loss_message)
                experiment.log(message)
                pbar.set_postfix(**loss_message)

        # histograms = {}
        # for tag, value in net.named_parameters():
        #     tag = tag.replace('/', '.')
        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
        #     # print(f'gradient: {value.grad.data.cpu()},'
        #     # print(f'\nMaxGradient: {value.grad.data.cpu().max()}'
        #     #       f'\nMinGradient: {value.grad.data.cpu().min()}')
        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        if save_checkpoint and (epoch + 1) % 20 == 0:
            logging.info('saving the latest model (epoch %d)' % (epoch + 1))
            model.save_networks(epoch + 1, dir_checkpoint)
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

    Discriminator = NLayerDiscriminator().to(device)
    Generator = UNet(n_channels=2, n_classes=1, bilinear=False).to(device)

    start_epoch = 0
    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     start_epoch = int(args.load.split('_')[-1].split('.')[0][5:])
    #     logging.info(f'Model loaded from {args.load}')

    train_net(discriminator=Discriminator,
              generator=Generator,
              epochs=args.epochs,
              dir_checkpoint=args.save_dir,
              batch_size=args.batch_size,
              device=device,
              std=args.std,
              val_percent=args.val / 100,
              start_epoch=start_epoch,
              )
    # except KeyboardInterrupt:
    #     torch.save(net.state_dict(), 'INTERRUPTED.pth')
    #     logging.info('Saved interrupt')
    #     sys.exit(0)
