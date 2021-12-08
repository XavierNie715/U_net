import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from scipy import ndimage

from utils.data_loading import BasicDataset
from unet import UNet, RelativeL2Error
from utils.utils import plot_img_and_mask

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--no_plot', '-n', action='store_true', help='Do not plot')

    return parser.parse_args()


def plot_and_save(OH_std, SVF_std, mask, mask_true_gs_std, val_error_plot, sv_name):
    fig, ax = plt.subplots(1, 5)
    ax = ax.flatten()

    ax[0].set_title('OH')
    ax0 = ax[0].imshow(np.squeeze(OH_std))

    ax[1].set_title('SVF')
    ax[1].set_yticks([])
    ax1 = ax[1].imshow(np.squeeze(SVF_std))

    ax[2].set_title('GT')
    ax[2].set_yticks([])
    ax2 = ax[2].imshow(mask_true_gs_std.reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0])

    ax[3].set_title('Pred')
    ax[3].set_yticks([])
    ax3 = ax[3].imshow(mask.reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0], cmap=cm.viridis)
    fig.colorbar(ax3, ax=ax[3])

    ax[4].set_title('L2_error')
    ax[4].set_yticks([])
    ax4 = ax[4].imshow(val_error_plot.cpu().numpy().reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0],
                       cmap=cm.Reds)

    cb1 = fig.colorbar(ax4, ax=ax[4])
    cb1.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb1.update_ticks()
    # plt.subplots_adjust(left=0.4, right=0.7, wspace=0.1) # for fig only have 2 plots

    # ax[2].imshow(img.cpu().numpy()[:, :, 0], cmap=cm.gray)
    # # ax3.set_title(f'Rel L2 error = {val_error.item()}')
    # ax[2].imshow(img.cpu().numpy()[:, :, 1], cmap=cm.jet)
    # ax3 = ax[2].imshow(val_error_plot.cpu().numpy().reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0], alpha=0.4)
    # fig.colorbar(ax3)

    '''
    trying to plot overlayed fig

    fig, ax = plt.subplots()
    OH_std = InstanceNorm(img[:, :, 0].reshape([1, 1, img.shape[0], img.shape[1]])).cpu().numpy()
    SVF_std = InstanceNorm(img[:, :, 1].reshape([1, 1, img.shape[0], img.shape[1]])).cpu().numpy()

    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkgreen"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkblue"])
    cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "red"])
    ax1 = ax.imshow(OH_std.reshape(img.shape[0], img.shape[1], -1), cmap=cmap1, interpolation='bilinear',
                    vmin=0, vmax=1)
    ax2 = ax.imshow(SVF_std.reshape(img.shape[0], img.shape[1], -1), cmap=cmap2, alpha=0.4,
                    interpolation='bilinear', vmin=0, vmax=1)

    # ax3 = ax.imshow(val_error_plot.cpu().numpy().reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0], cmap=cmap3, alpha=0.4,
    #           interpolation='bilinear')
    # fig.colorbar(ax3)
    '''

    # plt.show()
    plt.savefig(sv_name + '.png', figsize=(24, 8), dpi=300, bbox_inches='tight')

    plt.close()


"""
CUDA_VISIBLE_DEVICES=1 srun -p gpu_v100 -w node9 python predict.py --model /public/home/lcc-dx07/UNet/checkpoints/SELU_noIN_lr1e-5_b1_mse_small/checkpoint_epoch18.pth --input ./data/val/ > $logpath/$savepath/out.txt 2>&1
"""

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    in_files = str(args.input)
    # print(in_files)
    sv_dir = os.path.dirname(args.model)
    out_dir = sv_dir + '/results_' + args.model.split('.')[-2].split('_')[-1]
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)

    net = UNet(n_channels=2, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_criterion = RelativeL2Error()
    MSE_criterion = nn.MSELoss()

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    img_num = 1
    val_error_total = 0
    MSE_error_total = 0
    RMSE_error_total = 0

    for filename in os.listdir(in_files):
        logging.info(f'\nProcessing image {img_num} / {len(os.listdir(in_files))} ...')

        img = torch.from_numpy(np.load(in_files + filename))
        img = img.to(device=device, dtype=torch.float32)
        input_data = img[:, :, :2].reshape(1, 2, img.shape[0], img.shape[1])
        mask_true = img[:, :, 3].reshape(1, 1, img.shape[0], img.shape[1])
        InstanceNorm = nn.InstanceNorm2d(1)
        mask_true_std = InstanceNorm(mask_true).cpu().numpy()
        mask_true_gs_std = ndimage.filters.gaussian_filter(mask_true_std.reshape([img.shape[0],
                                                                                  img.shape[1],
                                                                                  1]),
                                                           sigma=20)

        net.eval()
        with torch.no_grad():
            mask = net(input_data).cpu().numpy()

        # print('mask: ', torch.from_numpy(mask).to(device).size())
        # print('mask_true_gs_std: ', torch.from_numpy(mask_true_gs_std).to(device).size())

        OH_std = InstanceNorm(img[:, :, 0].reshape([1, 1, img.shape[0], img.shape[1]])).cpu().numpy()
        SVF_std = InstanceNorm(img[:, :, 1].reshape([1, 1, img.shape[0], img.shape[1]])).cpu().numpy()

        val_error_plot = val_criterion(torch.from_numpy(mask).to(device),
                                       torch.from_numpy(mask_true_gs_std).to(device).reshape(1, -1,
                                                                                             mask.shape[2],
                                                                                             mask.shape[3]),
                                       reduct='none')
        val_error = val_error_plot.mean()
        MSE_error = MSE_criterion(torch.from_numpy(mask).to(device),
                                  torch.from_numpy(mask_true_gs_std).to(device).reshape(1, -1,
                                                                                        mask.shape[2],
                                                                                        mask.shape[3]), )
        RMSE_error = MSE_error.sqrt()

        val_error_total += val_error.item()
        MSE_error_total += MSE_error.item()
        RMSE_error_total += RMSE_error.item()

        sv_name = out_dir + '/' + filename.split('/')[-1].split('.')[0]
        # np.save(sv_name + '.npy', mask)

        logging.info(f'\n{filename} saved!\n'
                     f'Rel_L2_error = {val_error}\n'
                     f'MSE_error = {MSE_error}\n'
                     f'RMSE_error = {RMSE_error}')

        if args.no_plot == False:
            plot_and_save(OH_std, SVF_std, mask, mask_true_gs_std, val_error_plot, sv_name)

        img_num += 1

    val_error_total_mean = val_error_total / len(os.listdir(in_files))
    MSE_error_total_mean = MSE_error_total / len(os.listdir(in_files))
    RMSE_error_total_mean = RMSE_error_total / len(os.listdir(in_files))

    logging.info(f'\nFinish predict!\n'
                 f'mean_L2_error = {val_error_total_mean}\n'
                 f'mean_MSE_error = {MSE_error_total_mean}\n'
                 f'mean_RMSE_error = {RMSE_error_total_mean}')
