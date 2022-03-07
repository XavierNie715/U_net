import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


def threshold_mask(data, threshold_value=0.2, radius=50):
    threshold = (data.max() - data.min()) * threshold_value
    mask = np.ones(data.shape)
    W = len(mask)
    H = len(mask[0])
    mask[mask * data < threshold] = 0
    mask = ndimage.filters.maximum_filter(mask, radius)
    return mask


def std_GS(data, GS=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.from_numpy(data).to(device=device, dtype=torch.float32).reshape(1, -1, 789, 113)
    InstanceNorm = nn.InstanceNorm2d(1)
    data_std = InstanceNorm(data).cpu().numpy()
    if GS == True:
        data_gs_std = ndimage.filters.gaussian_filter(data_std.reshape([789, 113, 1]), sigma=20)
        return data_gs_std
    else:
        return data_std


class RelativeL2Error(nn.Module):
    """
    input: tensor, output: single tensor value
    if plot=Ture, then output: TENSOR (default plot=Faulse)
    f: regressed value, g: reference value
    Calculate relative l2 error between result and ground truth, see HFM paper
    """

    def __init__(self):
        super().__init__()

    def forward(self, f, g, reduct='mean'):
        return F.mse_loss(f, g, reduction=reduct) / F.mse_loss(g, torch.mean(g))


def temp_recover(temp_ori, temp_pred):
    """
    Recover temperature from standard deviation and mean.
    """
    temp_ori = temp_ori.reshape(1, -1, 789, 113)
    std_ori = np.std(temp_ori)
    mean_ori = np.mean(temp_ori)

    temp_real = temp_pred * std_ori + mean_ori

    return temp_real


def cal_PSNR(mse, img_true):
    if mse == 0:
        return 100
    return 10 * np.log10(img_true.max() ** 2 / mse)


def cal_SSIM(img_pred, img_true):
    assert img_pred.shape == img_true.shape
    mu1 = img_pred.mean()
    mu2 = img_true.mean()
    sigma1 = np.std(img_pred)
    sigma2 = np.std(img_true)
    sigma12 = ((img_pred - mu1) * (img_true - mu2)).mean()
    k1, k2 = 0.01, 0.03
    L = img_true.max()
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim
