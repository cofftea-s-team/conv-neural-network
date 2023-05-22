#pragma once
#include "../../cnn/config.hpp"
#include "../algebra/utils.hpp"
#include <cmath>

namespace host::optimizers {

	namespace parallel {

		template <class _Ty, class _Params>
		inline void adam_update(_Ty* _Weights, const _Ty* _Gradients, _Ty* _F_m, _Ty* _S_m, size_t N, size_t M,
			_Params _Hp, _Ty _Current_lr, _Ty _Current_beta1, _Ty _Current_beta2)
		{
			algebra::parallel::parallel_for(M, [=](size_t i) {
				for (size_t j = 0; j < N; ++j) {
					size_t _Idx = i * N + j;
					_F_m[_Idx] = _Hp.beta1 * _F_m[_Idx] + (1.0 - _Hp.beta1) * _Gradients[_Idx];
					_S_m[_Idx] = _Hp.beta2 * _S_m[_Idx] + (1.0 - _Hp.beta2) * (_Gradients[_Idx] * _Gradients[_Idx]);

					_Ty _F_m_hat = _F_m[_Idx] / (1.0 - _Current_beta1);
					_Ty _S_m_hat = _S_m[_Idx] / (1.0 - _Current_beta2);
					_Weights[_Idx] -= _Current_lr * (_F_m_hat / (std::sqrt(_S_m_hat) + _Hp.epsilon));
				}
			});
		}

	}

	template <class _Ty, class _Params>
	inline void adam_update(_Ty* _Weights, const _Ty* _Gradients, _Ty* _F_m, _Ty* _S_m, size_t _Size,
		_Params _Hp, _Ty _Current_lr, _Ty _Current_beta1, _Ty _Current_beta2)
	{
		for (size_t i = 0; i < _Size; ++i) {
			_F_m[i] = _Hp.beta1 * _F_m[i] + (1.0 - _Hp.beta1) * _Gradients[i];
			_S_m[i] = _Hp.beta2 * _S_m[i] + (1.0 - _Hp.beta2) * (_Gradients[i] * _Gradients[i]);

			_Ty _F_m_hat = _F_m[i] / (1.0 - _Current_beta1);
			_Ty _S_m_hat = _S_m[i] / (1.0 - _Current_beta2);

			_Weights[i] -= _Current_lr * (_F_m_hat / (std::sqrt(_S_m_hat) + _Hp.epsilon));
			
			if (std::isnan(_Weights[i])) {
				_Weights[i] = 0.0;
			}
		}
	}

}

namespace host {
	
	template <cnn::matrix_t _Mat1, cnn::matrix_t _Mat2, class _Ty, class _Params>
	inline void adam_update(_Mat1& _Weights, const _Mat1& _Gradients, _Mat2& _F_m, _Mat2& _S_m, 
		const _Params& _Hp, _Ty _Current_lr, _Ty _Current_beta1, _Ty _Current_beta2)
	{
		size_t N = _Weights.cols();
		size_t M = _Weights.rows();
		
		if (N * M < 2048) {
			host::optimizers::adam_update(_Weights.data(), _Gradients.data(), _F_m.data(), _S_m.data(), N * M, _Hp, _Current_lr, _Current_beta1, _Current_beta2);
		}
		else {
			host::optimizers::parallel::adam_update(_Weights.data(), _Gradients.data(), _F_m.data(), _S_m.data(), N, M, _Hp, _Current_lr, _Current_beta1, _Current_beta2);
		}
	}
}
/*
		template <matrix_t _Mat>
		inline void step(_Mat& _Tensor, const _Mat& _Gradient) {

			auto _Gradient_squared_hat = _Gradient * _Gradient / (1.0 - (_Current_beta2));
			auto _Gradient_hat = _Gradient / (1.0 - (_Current_beta1));
			_Gradient_squared_hat.activate<cnn::sqrt>();

			_Tensor += (_Gradient_hat / (_Gradient_squared_hat + _Hp.epsilon)) * _Lr;
			++_Iterations;
		}
*/