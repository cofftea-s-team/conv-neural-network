#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "../memory.hpp"
#include "../utils_cuda.cuh"
using bfloat16 = nv_bfloat16;

namespace cuda {

	template <class _Ty>
	_Ty _range_max(const _Ty* _Data, size_t N);

	template <class _Mat>
	inline auto range_max(const _Mat& _Data) {
		size_t N = _Data.cols();
		size_t M = _Data.rows();

		return _range_reduce(_Data.data(), _Data.size());
	}
}