import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

# n_groups defines block size as blockDim.x = n_groups * n_elements
#   -> Allowed values (int): 0 < n_groups < 6
#   -> Best result were achieved with n_groups = 2 (for image sizes of 300x300)
def delay_and_sum_cuda(rf_data_in, delays_idx, apod_weights, n_groups):
    
    # Source module
    module = SourceModule("""
        #include <cuComplex.h>

        #define NUM_ELEMENTS 192

        __global__ void delay_and_sum_cuda( cuFloatComplex *rf_data_in, float *delays_idx, float *apod_weights, cuFloatComplex *das_out, 
                                            int rf_data_size, int num_points, int num_groups, int N)
        {  
            const unsigned int jk_idx = threadIdx.x + blockIdx.x * blockDim.x;

            if(jk_idx < N){
        
                extern __shared__ cuFloatComplex s_das_out[];
                
                const int j = jk_idx/NUM_ELEMENTS;
                const int k = jk_idx - j * NUM_ELEMENTS;
                
                const int kj_idx = j + k * num_points;
                
                const int rf_data_idx = k + (int)delays_idx[ kj_idx ] * NUM_ELEMENTS;
                
                unsigned int group_id = threadIdx.x / NUM_ELEMENTS;
                unsigned int offset = group_id * NUM_ELEMENTS;
                unsigned int idx_off = threadIdx.x - offset;
                
                if ( rf_data_idx < rf_data_size)
                {
                    cuFloatComplex rf_data_idxd = rf_data_in[ rf_data_idx ];
                    cuFloatComplex complex_apod_weight_idxd = make_cuFloatComplex( apod_weights[ jk_idx ] , 0 );
                    s_das_out[threadIdx.x] = cuCmulf( rf_data_idxd , complex_apod_weight_idxd );
                }
                else s_das_out[threadIdx.x] = make_cuFloatComplex(0,0);

                __syncthreads();

                if (idx_off < 64)
                {
                    s_das_out[threadIdx.x] = cuCaddf( s_das_out[threadIdx.x] , s_das_out[threadIdx.x + 128] );
                    s_das_out[threadIdx.x] = cuCaddf( s_das_out[threadIdx.x] , s_das_out[threadIdx.x + 64] );
                    __syncthreads();
                }
                if (idx_off < 32)
                {
                    s_das_out[threadIdx.x]  = cuCaddf( s_das_out[threadIdx.x] ,  s_das_out[threadIdx.x + 32] );
                    s_das_out[threadIdx.x]  = cuCaddf( s_das_out[threadIdx.x] ,  s_das_out[threadIdx.x + 16] );
                    s_das_out[threadIdx.x]  = cuCaddf( s_das_out[threadIdx.x] ,  s_das_out[threadIdx.x + 8] );
                    s_das_out[threadIdx.x]  = cuCaddf( s_das_out[threadIdx.x] ,  s_das_out[threadIdx.x + 4] ); 
                    s_das_out[threadIdx.x]  = cuCaddf( s_das_out[threadIdx.x] ,  s_das_out[threadIdx.x + 2] ); 
                    s_das_out[threadIdx.x]  = cuCaddf( s_das_out[threadIdx.x] ,  s_das_out[threadIdx.x + 1] );
                }
                if (idx_off == 0)
                {
                    das_out[blockIdx.x * num_groups + group_id] = s_das_out[threadIdx.x];
                }
            }
        }
    """)

    # Constants
    n_elements = rf_data_in.shape[1]
    n_modes = delays_idx.shape[0]
    n_points = delays_idx.shape[2]

    # Allocate array on host
    das_out = np.zeros((n_modes, n_points), dtype=np.complex64, order='C')

    # Allocate device memory
    d_das_out = gpuarray.empty(shape=das_out.shape[1], dtype=np.complex64)          # Only partial array needed for each iteration
    
    # Flatten matrices
    rf_data_in = rf_data_in.flatten(order='C')
    apod_weights = apod_weights.flatten(order='C')

    # Allocate device memory and copy data to device
    #   -> delays_idx is copied within the for loop
    d_rf_data = gpuarray.to_gpu(rf_data_in.astype(np.complex64))
    d_apod_weights = gpuarray.to_gpu(apod_weights.astype(np.float32))

    # Constants required for kernel call
    N = n_elements * n_points                           # Number of valid threads
    n_threads = int(n_groups * n_elements)              # Number of threads per block
    n_blocks = int((N + n_threads - 1)/n_threads)       # Number of thread blocks
    sm_size = 8 * n_threads                             # Size of shared memory = 8 (bytes) * block dimension

    # Get kernel from module
    kernel = module.get_function("delay_and_sum_cuda")

    for i in range(n_modes):
        # Flatten and copy partial delays_idx array to allocated device memory
        delays_idx_i = delays_idx[i,:,:].flatten(order='C')
        d_delays_idx = gpuarray.to_gpu(delays_idx_i.astype(np.float32))
        # Kernel call
        kernel(d_rf_data, d_delays_idx, d_apod_weights, d_das_out, 
               np.uint32(rf_data_in.nbytes), 
               np.uint32(n_points), 
               np.uint32(n_groups), 
               np.uint32(N), 
               grid = (n_blocks, 1), 
               block = (n_threads, 1, 1),
               shared = sm_size)
        # Copy partial result back to host
        # cuda.memcpy_dtoh(d_das_out, das_out[i,:])
        das_out[i,:] = d_das_out.get()

    return das_out
