#include "matrix_mul_add_bias.cuh"


#define TILE_SIZE 32


template <class _Ty>
__global__ void matrix_mul_add_bias_kernel(const _Ty* A, const _Ty* B, const _Ty* V, _Ty* C, size_t M1, size_t N, size_t M2)
{
    __shared__ _Ty _Shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ _Ty _Shared_B[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    _Ty _Sum = 0.0;

    for (size_t i = 0; i < (N - 1) / TILE_SIZE + 1; ++i) {
        if (row < M1 && i * TILE_SIZE + threadIdx.x < N) {
             _Shared_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x];
        }
        else {
            _Shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < M2 && i * TILE_SIZE + threadIdx.y < N) {
            _Shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * M2 + col];
        }
        else {
            _Shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (size_t j = 0; j < TILE_SIZE; ++j) {
            _Sum = _Sum + _Shared_A[threadIdx.y][j] * _Shared_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M1 && col < M2) {
        C[row * M2 + col] = _Sum + V[col];
    }
}

template <class _Ty>
inline void matrix_shared_mul_add_bias(const _Ty* A, const _Ty* B, const _Ty* V, _Ty* C, size_t N, size_t M1, size_t M2)
{
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks((M2 - 1) / TILE_SIZE + 1, (M1 - 1) / TILE_SIZE + 1);

    matrix_mul_add_bias_kernel
        <<<blocks, threads>>>(A, B, V, C, M1, N, M2);
}

namespace cuda {

    template <class _Ty>
    void _matrix_mul_add_bias(const _Ty* A, const _Ty* B, const _Ty* V, _Ty* C, size_t N, size_t M1, size_t M2) {
        matrix_shared_mul_add_bias(A, B, V, C, N, M1, M2);
    }
    
#define INSTANTIATE_ONE(_Type) \
	template void _matrix_mul_add_bias<_Type>(const _Type*, const _Type*, const _Type*, _Type*, size_t, size_t, size_t);

	INSTANTIATE_ONE(float);
	INSTANTIATE_ONE(double);
	INSTANTIATE_ONE(bfloat16);
}
