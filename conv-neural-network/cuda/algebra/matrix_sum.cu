#include "matrix_sum.cuh"

#define BLOCK_DIM 32

namespace cuda {

	template <bool _T1, bool _T2, class _Ty>
	void _matrix_sum(const _Ty* A, _Ty* V, size_t N, size_t M) {
		for (size_t i = 0; i < M; ++i) {
			_Ty _S = cuda::_range_reduce(&A[i * N], N, 1);
			cuda::memcpy(&_S, &V[i], 1, HostToDevice);
		}
	}

#define INSTANTIATE(_Ty) \
	template void _matrix_sum<false, false, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum<false, true, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum<true, false, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum<true, true, _Ty>(const _Ty*, _Ty*, size_t, size_t);

	INSTANTIATE(double);
	INSTANTIATE(float);
	INSTANTIATE(bfloat16);
}