import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import threshold_mask, RelativeL2Error, cal_SSIM, cal_PSNR, temp_recover

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


def plot_and_save(filename, OH, SVF, T_pred, T_true_gs, sv_name):
    fig, ax = plt.subplots(nrows=5, ncols=1, constrained_layout=True, dpi=300)
    ax = ax.flatten()

    if filename.split('_')[0] == '220mm':
        x_Lf = '0.67'
        y_labels = ['251', '246', '241']

    elif filename.split('_')[0] == '245mm':
        x_Lf = '0.77'
        y_labels = ['286', '281', '276']

    else:
        x_Lf = '0.82'
        y_labels = ['306', '301', '296']

    x_ticks = [55, 168, 281, 394, 507, 620, 733]
    x_labels = ['-30', '-20', '-10', '0', '10', '20', '30']
    y_ticks = [0, 56, 112]

    fig.suptitle(filename.split('.')[0] + ' (x/Lf: ' + x_Lf + ')', fontsize=6, fontweight='bold')

    sub0 = ax[0].imshow(OH.T, cmap=cm.jet)
    ax[0].set_title('OH', fontsize=8)
    ax[0].set_xticks(ticks=x_ticks)
    ax[0].set_xticklabels(labels=x_labels, fontsize=6)
    ax[0].set_yticks(ticks=y_ticks)
    ax[0].set_yticklabels(labels=y_labels, fontsize=6)
    sub0.set_clim(0, OH.max())

    sub1 = ax[1].imshow(SVF.T, cmap=cm.jet)
    ax[1].set_title('SVF', fontsize=8)
    ax[1].set_xticks(ticks=x_ticks)
    ax[1].set_xticklabels(labels=x_labels, fontsize=6)
    ax[1].set_yticks(ticks=y_ticks)
    ax[1].set_yticklabels(labels=y_labels, fontsize=6)
    sub1.set_clim(0, SVF.max())

    sub2 = ax[2].imshow(T_pred.reshape([789, 113]).T, cmap=cm.jet)
    ax[2].set_title('T (pred.)', fontsize=8)
    ax[2].set_xticks(ticks=x_ticks)
    ax[2].set_xticklabels(labels=x_labels, fontsize=6)
    ax[2].set_yticks(ticks=y_ticks)
    ax[2].set_yticklabels(labels=y_labels, fontsize=6)
    sub2.set_clim(500, 2000)

    sub3 = ax[3].imshow(T_true_gs.reshape([789, 113]).T, cmap=cm.jet)
    ax[3].set_title('T (exp.)', fontsize=8)
    ax[3].set_xlabel('r [mm]', fontsize=6)
    ax[3].set_xticks(ticks=x_ticks)
    ax[3].set_xticklabels(labels=x_labels, fontsize=6)
    ax[3].set_ylabel('x [mm]', fontsize=6)
    ax[3].set_yticks(ticks=y_ticks)
    ax[3].set_yticklabels(labels=y_labels, fontsize=6)
    sub3.set_clim(500, 2000)

    cb1 = fig.colorbar(sub1, ax=ax[:2], ticks=[0, SVF.max()])
    cb1.set_ticklabels(['0', 'max'])
    cb1.ax.tick_params(labelsize=6)
    cb2 = fig.colorbar(sub3, ax=[ax[2], ax[3]])
    cb2.set_ticks([500, 1000, 1500, 2000])
    cb2.set_ticklabels(['500', '1000', '1500', '2000'])
    cb2.ax.tick_params(labelsize=6)

    x = np.linspace(1, 789, 789)
    sub4 = ax[4].plot(x, T_pred.reshape([789, 113]).T[55, :], 'r-', label='T (pred.)')
    sub5 = ax[4].plot(x, T_true_gs.reshape([789, 113]).T[55, :], 'b-', label='T (exp.)')
    plt.legend()

    # plt.show()
    plt.savefig(sv_name + '.png', dpi=300)
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
    out_dir = sv_dir + '/real_temp_results_' + args.model.split('.')[-2].split('_')[-1]
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)

    net = UNet(n_channels=2, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L2_criterion = RelativeL2Error()
    MSE_criterion = nn.MSELoss()

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    img_num = 1
    L2_error_total = 0
    L2_error_mask_total = 0

    MSE_error_total = 0
    MSE_error_mask_total = 0

    RMSE_error_total = 0
    RMSE_error_mask_total = 0

    SSIM_error_total = 0
    SSIM_error_mask_total = 0

    PSNR_error_total = 0
    PSNR_error_mask_total = 0

    for filename in os.listdir(in_files):
        logging.info(f'\nProcessing image {img_num} / {len(os.listdir(in_files))} ...')
        data = np.load(in_files + filename)

        OH = data[:, :, 0]
        SVF = data[:, :, 1]
        T = data[:, :, 3].reshape(-1, 1, data.shape[0], data.shape[1])

        mask = threshold_mask(OH) + threshold_mask(SVF)
        mask[mask > 0] = 1
        mask = torch.tensor(mask).to(device)

        img = torch.from_numpy(data)
        img = img.to(device=device, dtype=torch.float32)
        input_data = img[:, :, :2].reshape(1, 2, img.shape[0], img.shape[1])
        T_true_gs = ndimage.filters.gaussian_filter(T.reshape([img.shape[0],
                                                               img.shape[1],
                                                               1]),
                                                    mode='nearest',
                                                    sigma=20).reshape([1, 1, img.shape[0], img.shape[1]])

        net.eval()
        with torch.no_grad():
            T_pred = temp_recover(T, net(input_data).cpu().numpy())

        # print('mask: ', torch.from_numpy(mask).to(device).size())
        # print('mask_true_gs_std: ', torch.from_numpy(mask_true_gs_std).to(device).size())
        InstanceNorm = nn.InstanceNorm2d(1)
        # OH_std = InstanceNorm(img[:, :, 0].reshape([1, 1, img.shape[0], img.shape[1]])).cpu().numpy()
        # SVF_std = InstanceNorm(img[:, :, 1].reshape([1, 1, img.shape[0], img.shape[1]])).cpu().numpy()

        L2_error_plot = L2_criterion(torch.from_numpy(T_pred).to(device),
                                     torch.from_numpy(T_true_gs).to(device),
                                     reduct='none')
        L2_error = L2_error_plot.mean()
        L2_mask_error = (L2_error_plot * mask).mean()
        L2_error_total += L2_error.item()
        L2_error_mask_total += L2_mask_error.item()

        MSE_error = MSE_criterion(torch.from_numpy(T_pred).to(device),
                                  torch.from_numpy(T_true_gs).to(device))
        MSE_mask_error = MSE_criterion(torch.from_numpy(T_pred).to(device) * mask,
                                       torch.from_numpy(T_true_gs).to(device) * mask)
        MSE_error_total += MSE_error.item()
        MSE_error_mask_total += MSE_mask_error.item()

        RMSE_error = MSE_error.sqrt()
        RMSE_mask_error = MSE_mask_error.sqrt()
        RMSE_error_total += RMSE_error.item()
        RMSE_error_mask_total += RMSE_mask_error.item()

        SSIM_error = cal_SSIM(T_pred, T_true_gs)
        SSIM_mask_error = cal_SSIM(T_pred * mask.cpu().numpy(), T_true_gs * mask.cpu().numpy())
        SSIM_error_total += SSIM_error.item()
        SSIM_error_mask_total += SSIM_mask_error.item()

        PSNR_error = cal_PSNR(MSE_error.item(), T_true_gs)
        PSNR_mask_error = cal_PSNR(MSE_mask_error.item(), T_true_gs)
        PSNR_error_total += PSNR_error.item()
        PSNR_error_mask_total += PSNR_mask_error.item()

        sv_name = out_dir + '/' + filename.split('/')[-1].split('.')[0]
        np.save(sv_name + '.npy', T_pred)

        logging.info(f'\n{filename} saved!\n'
                     f'Rel_L2_error = {L2_error}\n'
                     f'Rel_L2_mask_error = {L2_mask_error}\n'
                     f'MSE_error = {MSE_error}\n'
                     f'MSE_mask_error = {MSE_mask_error}\n'
                     f'RMSE_error = {RMSE_error}\n'
                     f'RMSE_mask_error = {RMSE_mask_error}\n'
                     f'SSIM_error = {SSIM_error}\n'
                     f'SSIM_mask_error = {SSIM_mask_error}\n'
                     f'PSNR_error = {PSNR_error}\n'
                     f'PSNR_mask_error = {PSNR_mask_error}\n')

        if args.no_plot == False:
            plot_and_save(filename, OH, SVF, T_pred, T_true_gs, sv_name)

        img_num += 1

        L2_error_total_mean = L2_error_total / len(os.listdir(in_files))
        L2_error_mask_total_mean = L2_error_mask_total / len(os.listdir(in_files))
        MSE_error_total_mean = MSE_error_total / len(os.listdir(in_files))
        MSE_error_mask_total_mean = MSE_error_mask_total / len(os.listdir(in_files))
        RMSE_error_total_mean = RMSE_error_total / len(os.listdir(in_files))
        RMSE_error_mask_total_mean = RMSE_error_mask_total / len(os.listdir(in_files))
        SSIM_error_total_mean = SSIM_error_total / len(os.listdir(in_files))
        SSIM_error_mask_total_mean = SSIM_error_mask_total / len(os.listdir(in_files))
        PSNR_error_total_mean = PSNR_error_total / len(os.listdir(in_files))
        PSNR_error_mask_total_mean = PSNR_error_mask_total / len(os.listdir(in_files))

    logging.info(f'\nFinish predict!\n'
                 f'mean_L2_error = {L2_error_total_mean}\n'
                 f'mean_L2_mask_error = {L2_error_mask_total_mean}\n'
                 f'mean_MSE_error = {MSE_error_total_mean}\n'
                 f'mean_MSE_mask_error = {MSE_error_mask_total_mean}\n'
                 f'mean_RMSE_error = {RMSE_error_total_mean}\n'
                 f'mean_RMSE_mask_error = {RMSE_error_mask_total_mean}\n'
                 f'mean_SSIM_error = {SSIM_error_total_mean}\n'
                 f'mean_SSIM_mask_error = {SSIM_error_mask_total_mean}\n'
                 f'mean_PSNR_error = {PSNR_error_total_mean}\n'
                 f'mean_PSNR_mask_error = {PSNR_error_mask_total_mean}\n')
