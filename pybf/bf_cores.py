import numpy as np
# from numba import jit

# Perform delay and sum operation
# Input: rf_data of shape (n_samples x n_elements)
# delays_idx of shape (n_modes x n_elements x n_points)
# @jit(parallel=True)
def delay_and_sum(rf_data, delays_idx, apod_weights = None):

    n_elements = rf_data.shape[1]
    n_modes = delays_idx.shape[0]
    n_points = delays_idx.shape[2]

    # Choose the right samples for each channel and point
    # using numpy fancy indexing

    # Create array for fancy indexing of channels
    # of size (n_modes x n_points x n_elements)
    # The last two dimensions are transposed to fit the rf_data format
    fancy_idx_channels = np.arange(0, n_elements)
    fancy_idx_channels = np.tile(fancy_idx_channels, (n_modes, n_points, 1))

    # Create array for fancy indexing of samples
    # of size (n_modes x n_points x n_elements)
    # The last two dimensions are transposed to fit the rf_data format  
    fancy_idx_samples = np.transpose(delays_idx, axes = [0, 2, 1])

    # Make the delay and sum operation by selecting the samples
    # using fancy indexing,
    # multiplying by apodization weights (optional)
    # and then summing them up along the last axis
    if apod_weights  is None:
        das_out = np.sum(rf_data[fancy_idx_samples, fancy_idx_channels], axis = -1)
    else:
        das_out = np.sum(np.multiply(rf_data[fancy_idx_samples, fancy_idx_channels], apod_weights), axis = -1)

    # Output shape: (n_modes x n_points)
    return das_out


