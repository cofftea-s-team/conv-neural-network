#include "matrix_scalar_mul.cuh"

#define BLOCK_DIM 32

namespace cuda {

	template <bool _T1, bool _T2, class _Ty>
	__global__ void matrix_scalar_mul_kernel(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M) {
		int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
		int j = blockIdx.x * BLOCK_DIM + threadIdx.x;

		if (i < N && j < M) {
			if constexpr (_T1 && _T2) {
				C[i * M + j] = A[j * N + i] * B[j * N + i];
			}
			else if constexpr (_T1 && !_T2) {
				C[i * M + j] = A[j * N + i] * B[i * M + j];
			}
			else if constexpr (!_T1 && _T2) {
				C[i * M + j] = A[i * M + j] * B[j * N + i];
			}
			else {
				C[i * M + j] = A[i * M + j] * B[i * M + j];
			}
		}
	}

	template <bool _T1, bool _T2, class _Ty>
	void _matrix_scalar_mul(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M) {
		const dim3 threads(BLOCK_DIM, BLOCK_DIM);
		const dim3 blocks((N - 1) / BLOCK_DIM + 1, (M - 1) / BLOCK_DIM + 1);

		matrix_scalar_mul_kernel<_T1, _T2>
			<< <blocks, threads >> > (A, B, C, N, M);
	}

	template void _matrix_scalar_mul<false, false, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_scalar_mul<true, false, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_scalar_mul<false, true, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_scalar_mul<true, true, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);

	template void _matrix_scalar_mul<false, false, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_scalar_mul<true, false, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_scalar_mul<false, true, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_scalar_mul<true, true, float>(const float*, const float*, float*, size_t, size_t);

	template void _matrix_scalar_mul<false, false, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_scalar_mul<true, false, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_scalar_mul<false, true, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_scalar_mul<true, true, double>(const double*, const double*, double*, size_t, size_t);
}