import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

import matplotlib.pyplot as plt


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()


    with torch.no_grad():
        output = net(input_data)

    return output.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=2, n_classes=1, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')

        img = torch.from_numpy(np.load(filename))
        img = img.to(device=device, dtype=torch.float32)
        input_data = img[:, :, :2].reshape(1, 2, img.shape[0], img.shape[1])
        mask_true = img[:, :, 3].reshape(1, 1, img.shape[0], img.shape[1])
        InstanceNorm = nn.InstanceNorm2d(1)
        mask_true = InstanceNorm(mask_true).cpu().numpy()

        mask = predict_img(net=net,
                           full_img=input_data,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        print(mask.shape)
        np.save('./result.npy', mask)

        ax1 = plt.subplot(1, 2, 1)
        plt.title('Pred')
        plt.imshow(mask.reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0])
        plt.colorbar()

        ax2 = plt.subplot(1, 2, 2)
        plt.title('GT')
        plt.imshow(mask_true.reshape(mask.shape[2], mask.shape[3], -1)[:, :, 0])
        plt.colorbar()

        plt.savefig("reslut.png", dpi=300)

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
