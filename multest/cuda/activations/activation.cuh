#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "../../activations.hpp"
//#include "../../utils.hpp"

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <class _Activation_class, class _Ty>
	void _activation_apply(_Ty* _Data, size_t N);

	template <class _Activation_class, class _Mat>
	inline void activation_apply(_Mat& _Matrix) {
		_activation_apply<_Activation_class>(_Matrix.data(), _Matrix.size());
	}
}