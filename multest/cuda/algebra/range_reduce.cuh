#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "../memory.hpp"

using bfloat16 = nv_bfloat16;

namespace cuda {

	template <class _Ty>
	_Ty _range_reduce(const _Ty* _Data, size_t N, size_t M);

	template <class _Mat>
	inline auto range_reduce(const _Mat& _Data) {
		using _Ty = typename _Mat::value_type;
		size_t N = _Data.cols();
		size_t M = _Data.rows();

		return _range_reduce(_Data.data(), N, M);
	}
}