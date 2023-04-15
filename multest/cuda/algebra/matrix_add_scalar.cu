#include "matrix_add_scalar.cuh"

#define BLOCK_DIM 32

namespace cuda {
	
	template <class _Ty>
	__global__ void matrix_add_scalar_kernel(const _Ty* A, _Ty* B, _Ty* C, size_t N, size_t M) {
		int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
		int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
		
		if (i < N && j < M) {
			B[i * M + j] = A[i * M + j] + *C;
		}
	}
	
	template <class _Ty>
	void _matrix_add_scalar(const _Ty* A, _Ty* B, _Ty _Val, size_t N, size_t M) {
		const dim3 threads(BLOCK_DIM, BLOCK_DIM);
		const dim3 blocks((N - 1) / BLOCK_DIM + 1, (M - 1) / BLOCK_DIM + 1);
		
		// move _Val to cuda
		_Ty* _Val_cuda;
		cudaMalloc(&_Val_cuda, sizeof(_Ty));
		cudaMemcpy((void*)_Val_cuda, (void*)&_Val, sizeof(_Ty), cudaMemcpyHostToDevice);
		
		matrix_add_scalar_kernel<<<blocks, threads>>>(A, B, _Val_cuda, N, M);

		cudaFree((void*)_Val_cuda);
	}

	template void _matrix_add_scalar<bfloat16>(const bfloat16*, bfloat16*, bfloat16, size_t, size_t);
	template void _matrix_add_scalar<float>(const float*, float*, float, size_t, size_t);
	template void _matrix_add_scalar<double>(const double*, double*, double, size_t, size_t);

}