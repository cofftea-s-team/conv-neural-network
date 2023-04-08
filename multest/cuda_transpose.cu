#include "cuda_transpose.cuh"

#define TILE_DIM 32
#define BLOCK_ROWS 256
namespace cuda {

    template <class _Ty>
    __global__ void transpose_kernel(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols)
    {
        __shared__ _Ty tile[TILE_DIM][TILE_DIM];

        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int index_in = y * _Cols + x;
        int index_out = x * _Rows + y;

        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            tile[threadIdx.y + i][threadIdx.x] = _Src[index_in + i * _Cols];
        }

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            _Dst[index_out + i * _Rows] = tile[threadIdx.x][threadIdx.y + i];
        }
    }


	template<class _Ty>
	void _matrix_copy_transposed(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols)
	{
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((_Rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_Cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

        transpose_kernel<<<numBlocks, threadsPerBlock >>>(_Src, _Dst, _Rows, _Cols);
	}
	
	template void _matrix_copy_transposed(const double* _Src, double* _Dst, size_t _Rows, size_t _Cols);
	template void _matrix_copy_transposed(const float* _Src, float* _Dst, size_t _Rows, size_t _Cols);
	template void _matrix_copy_transposed(const bfloat16* _Src, bfloat16* _Dst, size_t _Rows, size_t _Cols);
}