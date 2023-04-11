#include "matrix_transpose.cuh"


namespace cuda {

    template <class _Ty>
    __global__ void transpose_kernel(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols)
    {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (i < _Rows && j < _Cols)
			_Dst[i * _Cols + j] = _Src[j * _Rows + i];
    }

	template<class _Ty>
	void _matrix_transpose(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols)
	{
        const dim3 threadsPerBlock(32, 32);
        const dim3 numBlocks((_Rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_Cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

        transpose_kernel<<<numBlocks, threadsPerBlock>>>(_Src, _Dst, _Rows, _Cols);
	}
	
	template void _matrix_transpose(const double* _Src, double* _Dst, size_t _Rows, size_t _Cols);
	template void _matrix_transpose(const float* _Src, float* _Dst, size_t _Rows, size_t _Cols);
	template void _matrix_transpose(const bfloat16* _Src, bfloat16* _Dst, size_t _Rows, size_t _Cols);
}