#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "curand_kernel.h"
#include "curand.h"
#include "../memory.hpp"

using bfloat16 = nv_bfloat16;

namespace cuda {
	
	template <class _Ty>
	void _apply_dropout(_Ty* _Data, size_t N, _Ty _Dropout_rate);

	template <class _Mat>
	inline void apply_dropout(_Mat& A, typename _Mat::value_type _Dropout_rate) {
		size_t N = A.size();
		_apply_dropout(A.data(), N, _Dropout_rate);
	}
}