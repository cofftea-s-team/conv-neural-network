#include "adam_update.cuh"

#define BLOCK_SIZE 512

namespace cuda {
	template <class _Ty>
	__global__ void adam_update_kernel(_Ty* _Weights, const _Ty* _Gradients, _Ty* _F_m, _Ty* _S_m, size_t N,
		_Ty _Current_beta1, _Ty _Current_beta2, _Ty _Beta1, _Ty _Beta2, _Ty _Epsilon, _Ty _Lr)
	{
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (i >= N) return;
		
		_F_m[i] = _Beta1 * _F_m[i] + (1.0 - _Beta1) * _Gradients[i];
		_S_m[i] = _Beta2 * _S_m[i] + (1.0 - _Beta2) * (_Gradients[i] * _Gradients[i]);

		_Ty _F_m_hat = _F_m[i] / (1.0 - _Current_beta1);
		_Ty _S_m_hat = _S_m[i] / (1.0 - _Current_beta2);
		
		if constexpr (!std::is_same_v<_Ty, bfloat16>) 
			_Weights[i] += _Lr * (_F_m_hat / (sqrt(_S_m_hat) + _Epsilon));
		else
			_Weights[i] = _Weights[i] + _Lr * (_F_m_hat / (sqrt(_S_m_hat) + _Epsilon));
	}

	template <class _Ty>
	void _adam_update(_Ty* _Weights, const _Ty* _Gradients, _Ty* _F_m, _Ty* _S_m, size_t N, 
		_Ty _Current_beta1, _Ty _Current_beta2, _Ty _Beta1, _Ty _Beta2, _Ty _Epsilon, _Ty _Lr)
	{
		dim3 threads(BLOCK_SIZE);
		dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
		
		adam_update_kernel<<<blocks, threads>>>(_Weights, _Gradients, _F_m, _S_m, N,
												_Current_beta1, _Current_beta2, _Beta1, 
												_Beta2, _Epsilon, _Lr);
	}

#define INSTANTIATE_ONE(_Ty) \
	template void _adam_update<_Ty>(_Ty*, const _Ty*, _Ty*, _Ty*, size_t, _Ty, _Ty, _Ty, _Ty, _Ty, _Ty);

	INSTANTIATE_ONE(float);
	INSTANTIATE_ONE(double);
	INSTANTIATE_ONE(bfloat16);
}

/*
		for (size_t i = 0; i < _Size; ++i) {
			_F_m[i] = _Hp.beta1 * _F_m[i] + (1.0 - _Hp.beta1) * _Gradients[i];
			_S_m[i] = _Hp.beta2 * _S_m[i] + (1.0 - _Hp.beta2) * (_Gradients[i] * _Gradients[i]);

			_Ty _F_m_hat = _F_m[i] / (1.0 - _Current_beta1);
			_Ty _S_m_hat = _S_m[i] / (1.0 - _Current_beta2);

			_Weights[i] -= _Current_lr * (_F_m_hat / (std::sqrt(_S_m_hat) + _Hp.epsilon));
		}
*/