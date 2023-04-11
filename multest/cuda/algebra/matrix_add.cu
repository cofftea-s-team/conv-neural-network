#include "matrix_add.cuh"

#define BLOCKDIM 32

namespace cuda {
	template <bool _T1, bool _T2, class _Ty>
	__global__ void matrix_add_kernel(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M) {
		int i = blockIdx.y * BLOCKDIM + threadIdx.y;
		int j = blockIdx.x * BLOCKDIM + threadIdx.x;
		
		if (i < N && j < M) {
			if constexpr (_T1 && _T2) {
				C[i * M + j] = A[j * N + i] + B[j * N + i];
			}
			else if constexpr (_T1 && !_T2) {
				C[i * M + j] = A[j * N + i] + B[i * M + j];
			}
			else if constexpr (!_T1 && _T2) {
				C[i * M + j] = A[i * M + j] + B[j * N + i];
			}
			else {
				C[i * M + j] = A[i * M + j] + B[i * M + j];
			}
		}
	}
	
	template <bool _T1, bool _T2, class _Ty>
	void _matrix_add(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M) {
		const dim3 blockDim(BLOCKDIM, BLOCKDIM);
		const dim3 gridDim((N - 1) / BLOCKDIM + 1, (M - 1) / BLOCKDIM + 1);

		matrix_add_kernel<_T1, _T2>
			<<<gridDim, blockDim>>>(A, B, C, N, M);
	}

	template void _matrix_add<false, false, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_add<true, false, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_add<false, true, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_add<true, true, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);

	template void _matrix_add<false, false, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_add<true, false, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_add<false, true, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_add<true, true, float>(const float*, const float*, float*, size_t, size_t);

	template void _matrix_add<false, false, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_add<true, false, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_add<false, true, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_add<true, true, double>(const double*, const double*, double*, size_t, size_t);
}