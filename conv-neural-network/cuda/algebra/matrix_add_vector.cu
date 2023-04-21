#include "matrix_add_vector.cuh"

#define BLOCK_DIM 32

namespace cuda {

	template <bool _T1, bool _T2, class _Ty>
	__global__ void matrix_add_vector_kernel(const _Ty* A, const _Ty* V, _Ty* B, size_t N, size_t M)
	{
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < M && j < N) {
			if constexpr (_T1 && _T2) {
				B[i * N + j] = A[j * M + i] + V[i];
			}
			else if constexpr (_T1 && !_T2) {
				B[i * N + j] = A[j * M + i] + V[j];
			}
			else if constexpr (!_T1 && _T2) {
				B[i * N + j] = A[i * N + j] + V[i];
			}
			else {
				B[i * N + j] = A[i * N + j] + V[j];
			}
		}
	}

	template <bool _T1, bool _T2, class _Ty>
	void _matrix_add_vector(const _Ty* _Src1, const _Ty* _Src2, _Ty* _Dst, size_t N, size_t M) {
		dim3 threads(BLOCK_DIM, BLOCK_DIM);
		dim3 blocks((N - 1) / BLOCK_DIM + 1, (M - 1) / BLOCK_DIM + 1);
		
		matrix_add_vector_kernel<_T1, _T2>
			<<<blocks, threads>>>(_Src1, _Src2, _Dst, N, M);
	}

	template void _matrix_add_vector<false, false, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_add_vector<false, true, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_add_vector<true, false, double>(const double*, const double*, double*, size_t, size_t);
	template void _matrix_add_vector<true, true, double>(const double*, const double*, double*, size_t, size_t);

	template void _matrix_add_vector<false, false, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_add_vector<false, true, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_add_vector<true, false, float>(const float*, const float*, float*, size_t, size_t);
	template void _matrix_add_vector<true, true, float>(const float*, const float*, float*, size_t, size_t);

	template void _matrix_add_vector<false, false, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_add_vector<false, true, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_add_vector<true, false, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
	template void _matrix_add_vector<true, true, bfloat16>(const bfloat16*, const bfloat16*, bfloat16*, size_t, size_t);
}