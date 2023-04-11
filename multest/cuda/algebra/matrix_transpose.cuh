#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <class _Ty>
	void _matrix_transpose(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols);
	 
	template <class _Mat, class _Mat2>
	inline void matrix_copy_transposed(const _Mat& _Src, _Mat2& _Dst) {
		_matrix_transpose(_Src.data(), _Dst.data(), _Src.rows(), _Src.cols());
	}
}