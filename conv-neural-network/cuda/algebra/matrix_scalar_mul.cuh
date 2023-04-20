#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <bool _T1, bool _T2, class _Ty>
	void _matrix_scalar_mul(const _Ty* _Src1, const _Ty* _Src2, _Ty* _Dst, size_t N, size_t M);

	template <class _Mat, class _Mat2, class _Mat3>
	inline void matrix_scalar_mul(const _Mat& _Src1, const _Mat2& _Src2, _Mat3& _Dst) {
		constexpr bool _T1 = _Src1.is_transposed();
		constexpr bool _T2 = _Src2.is_transposed();

		size_t N = _Src1.cols();
		size_t M = _Src1.rows(); 

		_matrix_scalar_mul<_T1, _T2>(_Src1.data(), _Src2.data(), _Dst.data(), N, M);
	}
}