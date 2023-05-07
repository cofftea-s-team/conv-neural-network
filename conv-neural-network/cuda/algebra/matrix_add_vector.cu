#include "matrix_add_vector.cuh"

#define BLOCK_DIM 32

namespace cuda {

	template <bool _T1, bool _T2, class _Ty, class _Pr>
	__global__ void matrix_add_vector_kernel(const _Ty* A, const _Ty* V, _Ty* B, size_t N, size_t M, _Pr _Pred)
	{
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < M && j < N) {
			if constexpr (_T1 && _T2) {
				B[i * N + j] = _Pred(A[j * M + i], V[j]);
			}
			else if constexpr (_T1 && !_T2) {
				B[i * N + j] = _Pred(A[j * M + i], V[i]);
			}
			else if constexpr (!_T1 && _T2) {
				B[i * N + j] = _Pred(A[i * N + j], V[j]);
			}
			else {
				B[i * N + j] = _Pred(A[i * N + j], V[i]);
			}
		}
	}

	template <bool _T1, bool _T2, class _Ty, class _Pr>
	void _matrix_add_vector(const _Ty* _Src1, const _Ty* _Src2, _Ty* _Dst, size_t N, size_t M, _Pr _Pred) {
		dim3 threads(BLOCK_DIM, BLOCK_DIM);
		dim3 blocks((N - 1) / BLOCK_DIM + 1, (M - 1) / BLOCK_DIM + 1);
		
		matrix_add_vector_kernel<_T1, _T2>
			<<<blocks, threads>>>(_Src1, _Src2, _Dst, N, M, _Pred);
	}

#define INSTANTIATE_HELPER(_T1, _T2, _Type) \
template void _matrix_add_vector<_T1, _T2, _Type, cuda::plus<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::plus<_Type>); \
template void _matrix_add_vector<_T1, _T2, _Type, cuda::minus<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::minus<_Type>); \
template void _matrix_add_vector<_T1, _T2, _Type, cuda::multiplies<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::multiplies<_Type>); \
template void _matrix_add_vector<_T1, _T2, _Type, cuda::divides<_Type>>(const _Type*, const _Type*, _Type*, size_t, size_t, cuda::divides<_Type>); \


#define INSTANTIATE_ONE(_Type) \
INSTANTIATE_HELPER(false, false, _Type); \
INSTANTIATE_HELPER(false, true, _Type); \
INSTANTIATE_HELPER(true, false, _Type); \
INSTANTIATE_HELPER(true, true, _Type);

	INSTANTIATE_ONE(float);
	INSTANTIATE_ONE(double);
	INSTANTIATE_ONE(bfloat16);
}