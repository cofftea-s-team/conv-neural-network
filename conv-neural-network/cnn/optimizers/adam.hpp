#pragma once
#include "../config.hpp"
#include "../../host/optimizers/adam_update.hpp"
#include "../../cuda/optimizers/adam_update.cuh"
#include <vector>

namespace cnn::optimizers {

	class adam;

	template <>
	struct hyperparameters<adam> {
		using value_type = typename cnn::config::value_type;
		
		value_type learning_rate = 1e-3;
		value_type beta1 = 0.9;
		value_type beta2 = 0.999;
		value_type decay = 1e-4;
		value_type epsilon = 1e-7;
	};

	class adam {
		using value_type = typename cnn::config::value_type;
		using matrix = typename cnn::config::matrix;
		using vector = typename cnn::config::vector;
	public:
		constexpr adam(size_t layer_count, hyperparameters<adam> hyperparams = {}) 
			: _Hp(std::move(hyperparams)), _Layer_count(layer_count), _Current_lr(hyperparams.learning_rate)
			, _Current_beta1(1.), _Current_beta2(1.), _Iterations(0)
		{ 
			_W_first_moments.reserve(layer_count);
			_W_second_moments.reserve(layer_count);
			_B_first_moments.reserve(layer_count);
			_B_second_moments.reserve(layer_count);
		}

		template <matrix_t _Mat, matrix_t _Mat2, matrix_t _Vec>
		inline void step(_Mat& _Weights, const _Mat2& _W_gradients, _Vec& _Biases, const _Vec& _B_gradients) {
			if (_W_first_moments.size() < _Layer_count) {
				_W_first_moments.emplace_back(_Weights.shape());
				_W_second_moments.emplace_back(_Weights.shape());
				_B_first_moments.emplace_back(_Biases.shape());
				_B_second_moments.emplace_back(_Biases.shape());
			}
			
			size_t _Index = _Iterations % _Layer_count;

			if (_Index == 0) {
				_Current_lr = _Hp.learning_rate * (1.0 / (1.0 + _Hp.decay * (2.0 * _Iterations / _Layer_count)));
				_Current_beta1 *= _Hp.beta1;
				_Current_beta2 *= _Hp.beta2;
			}
			++_Iterations;

			if constexpr (config::device == device_type::cpu) {
				host::adam_update(_Weights, _W_gradients, _W_first_moments[_Index], _W_second_moments[_Index], _Hp, _Current_lr, _Current_beta1, _Current_beta2);
				host::adam_update(_Biases, _B_gradients, _B_first_moments[_Index], _B_second_moments[_Index], _Hp, _Current_lr, _Current_beta1, _Current_beta2);
			}
			else {
				cuda::adam_update(_Weights, _W_gradients, _W_first_moments[_Index], _W_second_moments[_Index], _Hp, _Current_lr, _Current_beta1, _Current_beta2);
				cuda::adam_update(_Biases, _B_gradients, _B_first_moments[_Index], _B_second_moments[_Index], _Hp, _Current_lr, _Current_beta1, _Current_beta2);
			}
		}

		inline friend std::ostream& operator<<(std::ostream& _Os, const adam& _Adam) {
			return _Os << "lr: " << _Adam._Current_lr << "  beta1: " << _Adam._Current_beta1
				<< "  beta2: " << _Adam._Current_beta2;
		}

	private:
		value_type _Current_lr;
		value_type _Current_beta1;
		value_type _Current_beta2;
		size_t _Iterations;
		std::vector<matrix> _W_first_moments;
		std::vector<matrix> _W_second_moments;
		std::vector<vector> _B_first_moments;
		std::vector<vector> _B_second_moments;
		
		const hyperparameters<adam> _Hp;
		const size_t _Layer_count;
	};
}