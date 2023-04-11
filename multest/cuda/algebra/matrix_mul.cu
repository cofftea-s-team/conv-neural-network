#include "matrix_mul.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32


template <bool _T1, bool _T2, class _Ty>
__global__ void matrix_mul_kernel(const _Ty* A, const _Ty* B, _Ty* C, size_t M1, size_t N, size_t M2)
{
    __shared__ _Ty _Shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ _Ty _Shared_B[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    _Ty _Sum = 0.0;

    for (size_t i = 0; i < (N - 1) / TILE_SIZE + 1; i++) {
        if (row < M1 && i * TILE_SIZE + threadIdx.x < N) {
            if constexpr (!_T1) 
                _Shared_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x];
            else 
                _Shared_A[threadIdx.y][threadIdx.x] = A[i * M1 + row * TILE_SIZE + threadIdx.x];
            
        }
        else {
            _Shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < M2 && i * TILE_SIZE + threadIdx.y < N) {
            if constexpr (!_T2) 
                _Shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * M2 + col];
            else 
                _Shared_B[threadIdx.y][threadIdx.x] = B[col * N + (i * TILE_SIZE + threadIdx.y)];
        }
        else {
            _Shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (size_t j = 0; j < TILE_SIZE; j++) {
            _Sum =  _Sum + _Shared_A[threadIdx.y][j] * _Shared_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M1 && col < M2) {
        C[row * M2 + col] = _Sum;
    }
}

template <bool _T1, bool _T2, class _Ty>
inline void matrix_shared_multiply(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M1, size_t M2)
{
    const dim3 blockDim(TILE_SIZE, TILE_SIZE);
    const dim3 gridDim((M2 - 1) / TILE_SIZE + 1, (M1 - 1) / TILE_SIZE + 1);

    matrix_mul_kernel<_T1, _T2><<<gridDim, blockDim>>>(A, B, C, M1, N, M2);
}

namespace cuda {
    
    template <class _Ty, bool _T1, bool _T2>
    void _matrix_multiply(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M1, size_t M2) {
        matrix_shared_multiply<_T1, _T2>(A, B, C, N, M1, M2);
    }
    
	template void _matrix_multiply<double, false, false>(const double*, const double*, double*, size_t, size_t, size_t);
	template void _matrix_multiply<double, false, true>(const double*, const double*, double*, size_t, size_t, size_t);
	template void _matrix_multiply<double, true, false>(const double*, const double*, double*, size_t, size_t, size_t);
	template void _matrix_multiply<double, true, true>(const double*, const double*, double*, size_t, size_t, size_t);
    
	template void _matrix_multiply<float, false, false>(const float*, const float*, float*, size_t, size_t, size_t);
	template void _matrix_multiply<float, false, true>(const float*, const float*, float*, size_t, size_t, size_t);
	template void _matrix_multiply<float, true, false>(const float*, const float*, float*, size_t, size_t, size_t);
	template void _matrix_multiply<float, true, true>(const float*, const float*, float*, size_t, size_t, size_t);
    
	template void _matrix_multiply<bfloat16, false, false>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t, size_t);
	template void _matrix_multiply<bfloat16, false, true>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t, size_t);
	template void _matrix_multiply<bfloat16, true, false>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t, size_t);
	template void _matrix_multiply<bfloat16, true, true>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t, size_t);
}
