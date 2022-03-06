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
