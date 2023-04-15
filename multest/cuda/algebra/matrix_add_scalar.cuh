#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <class _Ty>
	void _matrix_add_scalar(const _Ty* _Src1, _Ty* _Dst, _Ty _Val, size_t N, size_t M);

	template <class _Mat, class _Mat2, class _Ty>
	inline void matrix_add_scalar(const _Mat& _Src1, _Mat2& _Dst, _Ty _Val) {
		size_t N = _Src1.cols();
		size_t M = _Src1.rows();

		_matrix_add_scalar(_Src1.data(), _Dst.data(), _Val, N, M);
	}
}