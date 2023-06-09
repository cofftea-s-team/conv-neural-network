#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "../utils.hpp"

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <bool _T1, bool _T2, class _Ty, class _Pr>
	void _matrix_add_vector(const _Ty* _Src1, const _Ty* _Src2, _Ty* _Dst, size_t N, size_t M, _Pr _Pred);

	template <class _Pred, class _Mat, class _Vec, class _Mat2>
	inline void matrix_add_vector(const _Mat& _Src1, const _Vec& _Src2, _Mat2& _Dst) {
		constexpr bool _T1 = _Src1.is_transposed();
		constexpr bool _T2 = _Src2.is_transposed();

		size_t N = _Src1.cols();
		size_t M = _Src1.rows();

		_matrix_add_vector<_T1, _T2>(_Src1.data(), _Src2.data(), _Dst.data(), N, M, _Pred{});
	}
}