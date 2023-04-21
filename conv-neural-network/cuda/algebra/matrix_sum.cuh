#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "range_reduce.cuh"

using bfloat16 = nv_bfloat16;

namespace cuda {
	
	template <bool _T1, bool _T2, class _Ty>
	void _matrix_sum(const _Ty* A, _Ty* V, size_t N, size_t M);

	template <class _Mat, class _Vec>
	inline void matrix_sum(const _Mat& A, _Vec& V) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = V.is_transposed();

		if constexpr (_T1 || _T2) {
			static_assert(std::_Always_false<_Mat>, "matrix_sum: transposed matrices are not supported");
		}

		size_t N = A.cols();
		size_t M = A.rows();
		
		_matrix_sum<_T1, _T2>(A.data(), V.data(), N, M);
	}
	
}