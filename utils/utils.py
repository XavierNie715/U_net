import numpy as np
from scipy import ndimage


def threshold_mask(data, threshold_value=0.2, radius=50):
    threshold = (data.max() - data.min()) * threshold_value
    mask = np.ones(data.shape)
    W = len(mask)
    H = len(mask[0])
    mask[mask * data < threshold] = 0
    mask = ndimage.filters.maximum_filter(mask, radius)
    return mask
