#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <class _Ty>
	void _matrix_fill(_Ty* _Src, size_t N, _Ty _Val);

	template <class _Mat, class _Ty>
	inline void matrix_fill(_Mat& _Src, const _Ty& _Val) {
		_matrix_fill(_Src.data(), _Src.size(), _Val);
	}
}