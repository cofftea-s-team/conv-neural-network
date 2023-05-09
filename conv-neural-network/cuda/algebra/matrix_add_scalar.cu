#include "matrix_add_scalar.cuh"

#define BLOCK_DIM 32

namespace cuda {
	
	template <class _Ty>
	__global__ void matrix_add_scalar_kernel(const _Ty* A, _Ty* B, _Ty C, size_t N, size_t M) {
		int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
		int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
		
		if (i < M && j < N) {
			B[i * N + j] = A[i * N + j] + C;
		}
	}
	
	template <class _Ty>
	void _matrix_add_scalar(const _Ty* A, _Ty* B, _Ty _Val, size_t N, size_t M) {
		const dim3 threads(BLOCK_DIM, BLOCK_DIM);
		const dim3 blocks((N - 1) / BLOCK_DIM + 1, (M - 1) / BLOCK_DIM + 1);
		
		matrix_add_scalar_kernel<<<blocks, threads>>>(A, B, _Val, N, M);
	}

	template void _matrix_add_scalar<bfloat16>(const bfloat16*, bfloat16*, bfloat16, size_t, size_t);
	template void _matrix_add_scalar<float>(const float*, float*, float, size_t, size_t);
	template void _matrix_add_scalar<double>(const double*, double*, double, size_t, size_t);

}