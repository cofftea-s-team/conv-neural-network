#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "../algebra/range_reduce.cuh"
#include "../algebra/range_max.cuh"
#include "../algebra/matrix_add_scalar.cuh"
#include "../../cnn/activations.hpp"
//#include "../../utils.hpp"

using bfloat16 = nv_bfloat16;

namespace cuda {
	
	template <class _Activation_class>
	struct forwarder {
		template <class _Ty, class... _TArgs>
		__device__ inline static _Ty apply(const _Ty& _Val, _TArgs... _Args) {
			return _Activation_class::forward(_Val, _Args...);
		}
	};

	template <class _Activation_class>
	struct backwarder {
		template <class _Ty, class... _TArgs>
		__device__ inline static _Ty apply(const _Ty& _Val, _TArgs... _Args) {
			return _Activation_class::backward(_Val, _Args...);
		}
	};

	template <class _Activation_class, class _Ty>
	void _forward_apply(_Ty* _Data, size_t N, size_t M);

	template <class _Activation_class, class _Ty>
	void _backward_apply(_Ty* _Data, size_t N, size_t M);

	template <class _Activation_class, class _Mat>
	inline void forward_apply(_Mat& _Matrix) {
		_forward_apply<_Activation_class>(_Matrix.data(), _Matrix.cols(), _Matrix.rows());
	}

	template <class _Activation_class, class _Mat>
	inline void backward_apply(_Mat& _Matrix) {
		_backward_apply<_Activation_class>(_Matrix.data(), _Matrix.cols(), _Matrix.rows());
	}
}