#pragma once
#include <vector>
#include "../config.hpp"

namespace cnn::optimizers {

	class adam {
		using value_type = typename cnn::config::value_type;
		using matrix = typename cnn::config::matrix;
	public:
		constexpr adam(size_t layer_count = 0, value_type lr = 1e-3, value_type decay = 1e-4, value_type epsilon = 1e-7, value_type beta1 = 0.9, value_type beta2 = 0.999) {
			_Lr = lr;
			_Decay = decay;
			_Epsilon = epsilon;
			_Beta1 = beta1;
			_Current_beta1 = 1;
			_Beta2 = beta2;
			_Current_beta2 = 1;
			_Iterations = 0;
			_Layer_count = layer_count;
			_Current_lr = _Lr * (1.0 / (1.0 + _Decay * (2.0 * _Iterations / _Layer_count)));
		}

		template <class _Mat1, class _Mat2>
		inline void step(_Mat1& _Tensor, const _Mat2& _Gradient) {
			++_Iterations;
			if (_Iterations % _Layer_count == 1) {
				_Current_lr = _Lr * (1.0 / (1.0 + _Decay * (2.0 * _Iterations / _Layer_count)));
				_Current_beta1 *= _Beta1;
				_Current_beta2 *= _Beta2;
			}
			auto _Gradient_squared = _Gradient * _Gradient;
			auto _Gradient_squared_hat = _Gradient_squared / (1.0 - (_Current_beta2));
			auto _Gradient_hat = _Gradient / (1.0 - (_Current_beta1));
			_Gradient_squared_hat.activate<cnn::sqrt>();

			_Tensor += (_Gradient_hat / (_Gradient_squared_hat + _Epsilon)) * _Current_lr;
		}

		inline value_type learning_rate() const {
			return _Current_lr;
		}

	private:
		size_t _Iterations;
		value_type _Current_lr;
		value_type _Lr;
		value_type _Decay;
		value_type _Epsilon;
		value_type _Beta1, _Current_beta1;
		value_type _Beta2, _Current_beta2;
		size_t _Layer_count;
	};
}