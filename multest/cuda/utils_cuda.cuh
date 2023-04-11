#pragma once
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "curand.h"
#include "curand_kernel.h"

using bfloat16 = nv_bfloat16;
enum RAND_MODE {
	UNIFORM,
	NORMAL
};
namespace cuda {
	

	template <RAND_MODE _Mode, class _Ty>
	void _fill_random(_Ty* _Data, size_t N);

	template <RAND_MODE _Mode, class _Mat>
	inline void fill_random(_Mat& _M) {
		_fill_random<_Mode>(_M.data(), _M.size());
	}
}