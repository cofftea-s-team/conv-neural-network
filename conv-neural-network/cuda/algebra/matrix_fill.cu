#include "matrix_fill.cuh"

#define BLOCK_SIZE 512

namespace cuda {
	
	template <class _Ty>
	__global__ void _matrix_fill_kernel(_Ty* _Data, size_t N, _Ty _Val) {
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			_Data[i] = _Val;
		}
	}

	template <class _Ty>
	void _matrix_fill(_Ty* _Data, size_t N, _Ty _Val) {
		dim3 threads(BLOCK_SIZE);
		dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

		_matrix_fill_kernel<<<blocks, threads>>>(_Data, N, _Val);
	}

#define INSTANTIATE(_Ty) \
	template void _matrix_fill<_Ty>(_Ty*, size_t, _Ty);
	
	INSTANTIATE(float);
	INSTANTIATE(double);
	INSTANTIATE(nv_bfloat16);
}