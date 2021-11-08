"""
   Copyright (C) 2020 ETH Zurich. All rights reserved.

   Author: Sergei Vostrikov, ETH Zurich

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
from numba import jit, prange

# Perform delay and sum operation with numba
# Input: rf_data_in of shape (n_samples x n_elements)
# delays_idx of shape (n_modes x n_elements x n_points)
# apod_weights of shape (n_points x n_elements)
@jit(nopython = True, parallel = True, nogil = True)
def delay_and_sum_numba(rf_data_in, delays_idx, apod_weights = None):

    n_elements = rf_data_in.shape[1]
    n_modes = delays_idx.shape[0]
    n_points = delays_idx.shape[2]

    # Allocate array
    das_out = np.zeros((n_modes, n_points), dtype=np.complex64)

    # Iterate over modes, points, elements
    for i in range(n_modes):
        for j in prange(n_points):
            for k in range(n_elements):
                if (delays_idx[i, k, j] <= rf_data_in.shape[0] - 1):
                    if apod_weights is None:
                        das_out[i, j] += rf_data_in[delays_idx[i, k, j], k]
                    else:     
                        das_out[i, j] += rf_data_in[delays_idx[i, k, j], k] * apod_weights[j, k]

    return das_out

# Perform delay and sum operation with numpy
# Input: rf_data_in of shape (n_samples x n_elements)
# delays_idx of shape (n_modes x n_elements x n_points)
def delay_and_sum_numpy(rf_data_in, delays_idx, apod_weights = None):

    n_elements = rf_data_in.shape[1]
    n_modes = delays_idx.shape[0]
    n_points = delays_idx.shape[2]

    # Add one zero sample for data array (in the end)
    rf_data_shape = rf_data_in.shape
    rf_data = np.zeros((rf_data_shape[0] + 1, rf_data_shape[1]), dtype=np.complex64)
    rf_data[:rf_data_shape[0],:rf_data_shape[1]] = rf_data_in

    # If delay index exceeds the input data array dimensions, 
    # write -1 (it will point to 0 element)
    delays_idx[delays_idx >= rf_data_shape[0] - 1] = -1

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



