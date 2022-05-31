import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils.utils import std_GS

from scipy import ndimage


class BasicDataset(Dataset):
    '''
    原函数
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    '''

    def __init__(self, data_dir: str, std: bool = True):  # eg. input: './data/220mm, ./data/245mm'
        self.std = std

        self.ids = []
        for file_dir in data_dir.split(','):
            file_dir += '/npy/'
            self.ids.extend((file_dir + data) for data in listdir(file_dir))  # 提取所有文件名
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    # @classmethod  # 输入单张图片，用PIL放缩，返回ndarray
    # def preprocess(cls, pil_img, scale, is_mask):
    #     w, h, _ = pil_img.shape
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    #     pil_img = pil_img.resize((newW, newH))
    #     img_ndarray = np.asarray(pil_img)
    #
    #     if img_ndarray.ndim == 2 and not is_mask:
    #         img_ndarray = img_ndarray[np.newaxis, ...]
    #     elif not is_mask:
    #         img_ndarray = img_ndarray.transpose((2, 0, 1))
    #
    #     if not is_mask:
    #         img_ndarray = img_ndarray / 255
    #
    #     return img_ndarray

    '''
    原函数
    @classmethod  # 返回PIL的图片对象
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
    '''

    '''
    原函数
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
    '''

    def __getitem__(self, idx):
        name = self.ids[idx]
        data = torch.as_tensor(np.load(name))
        T = data[:, :, 3]
        T = std_GS(T, self.std)
        # N,C,H,W
        # data.shape[0]: 789, data.shape[1]: 113
        # InstanceNorm = nn.InstanceNorm2d(1)
        # T_std = InstanceNorm(T).cpu().numpy()
        # T_gs_std = ndimage.filters.gaussian_filter(T_std.reshape([data.shape[0],
        #                                                           data.shape[1],
        #                                                           1]),
        #                                            sigma=20)

        return {
            'image': data[:, :, :2].reshape(2, data.shape[0], data.shape[1]),  # only take OH and SVF as input
            'mask': T.reshape(-1, data.shape[0], data.shape[1])  # T
        }

        # class CarvanaDataset(BasicDataset):
        #     def __init__(self, images_dir, masks_dir, scale=1):
        #         super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
