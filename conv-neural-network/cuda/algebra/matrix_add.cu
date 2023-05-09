#include "matrix_add.cuh"

#define BLOCK_DIM 32

namespace cuda {
	
	template <bool _T1, bool _T2, class _Ty, class _Pr>
	__global__ void matrix_add_kernel(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M, _Pr _Pred) {
		int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
		int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
		
		if (i < M && j < N) { 
			if constexpr (_T1 && _T2) {
				C[i * N + j] = _Pred(A[j * M + i], B[j * M + i]);
			}
			else if constexpr (_T1 && !_T2) {
				C[i * N + j] = _Pred(A[j * M + i], B[i * N + j]);
			}
			else if constexpr (!_T1 && _T2) {
				C[i * N + j] = _Pred(A[i * N + j], B[j * M + i]);
			}
			else {
				C[i * N + j] = _Pred(A[i * N + j], B[i * N + j]);
			}
		}
	}
	
	template <bool _T1, bool _T2, class _Ty, class _Pr>
	void _matrix_add(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M, _Pr _Pred) {
		const dim3 threads(BLOCK_DIM, BLOCK_DIM);
		const dim3 blocks((N - 1) / BLOCK_DIM + 1, (M - 1) / BLOCK_DIM + 1);

		matrix_add_kernel<_T1, _T2>
			<<<blocks, threads>>>(A, B, C, N, M, _Pred);
	}
	
#define INSTANTIATE_HELPER(_T1, _T2, _Type) \
template void _matrix_add<_T1, _T2, _Type, cuda::plus<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::plus<_Type>); \
template void _matrix_add<_T1, _T2, _Type, cuda::minus<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::minus<_Type>); \
template void _matrix_add<_T1, _T2, _Type, cuda::multiplies<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::multiplies<_Type>); \
template void _matrix_add<_T1, _T2, _Type, cuda::divides<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::divides<_Type>); \


#define INSTANTIATE_ONE(_Type) \
INSTANTIATE_HELPER(false, false, _Type); \
INSTANTIATE_HELPER(false, true, _Type); \
INSTANTIATE_HELPER(true, false, _Type); \
INSTANTIATE_HELPER(true, true, _Type);

	INSTANTIATE_ONE(float);
	INSTANTIATE_ONE(double);
	INSTANTIATE_ONE(bfloat16);
}