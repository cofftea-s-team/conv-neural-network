#include "matrix_transpose.cuh"

#define BLOCK_DIM 32

namespace cuda {

    template <class _Ty>
    __global__ void transpose_kernel(const _Ty* _Src, _Ty* _Dst, size_t N, size_t M)
    {
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (i < N && j < M)
			_Dst[j * N + i] = _Src[i * M + j];
    }

	template<class _Ty>
	void _matrix_transpose(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols)
	{
		const dim3 threads(BLOCK_DIM, BLOCK_DIM);
		const dim3 blocks((_Cols - 1) / BLOCK_DIM + 1, (_Rows - 1) / BLOCK_DIM + 1);

        transpose_kernel<<<blocks, threads>>>(_Src, _Dst, _Cols, _Rows);
	}
	
	template void _matrix_transpose(const double* _Src, double* _Dst, size_t _Rows, size_t _Cols);
	template void _matrix_transpose(const float* _Src, float* _Dst, size_t _Rows, size_t _Cols);
	template void _matrix_transpose(const bfloat16* _Src, bfloat16* _Dst, size_t _Rows, size_t _Cols);
}