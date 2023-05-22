#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include <cmath>

using bfloat16 = nv_bfloat16;

namespace cuda {
	
	template <class _Ty>
	void _adam_update(_Ty* _Weights, const _Ty* _Gradients, _Ty* _F_m, _Ty* _S_m, size_t N,
		_Ty _Current_beta1, _Ty _Current_beta2, _Ty _Beta1, _Ty _Beta2, _Ty _Epsilon, _Ty _Lr);

	template <class _Mat, class _Ty, class _Params>
	inline void adam_update(_Mat& _Weights, const _Mat& _Gradients, _Mat& _F_m, _Mat& _S_m,
		const _Params& _Hp, _Ty _Current_lr, _Ty _Current_beta1, _Ty _Current_beta2) 
	{
		_adam_update(_Weights.data(), _Gradients.data(), _F_m.data(), _S_m.data(), _Weights.size(), _Current_beta1, _Current_beta2, _Hp.beta1, _Hp.beta2, _Hp.epsilon, _Current_lr);
	}
}