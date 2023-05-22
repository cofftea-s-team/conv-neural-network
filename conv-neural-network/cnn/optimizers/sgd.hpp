#pragma once
#include "../config.hpp"

namespace cnn::optimizers {
	
	class sgd;

	template <>
	struct hyperparameters<sgd> {
		using value_type = typename cnn::config::value_type;
		
		constexpr hyperparameters(value_type _Lr = 1e-4)
			: learning_rate(_Lr)
		{ }

		value_type learning_rate;
	};

	class sgd {
		using value_type = typename cnn::config::value_type;
		using matrix = typename cnn::config::matrix;
	public:
		constexpr sgd(size_t layer_count, hyperparameters<sgd> hyperparams = {})
			: _Hp(std::move(hyperparams)), _Layer_count(layer_count)
		{ }

		template <matrix_t _Mat, matrix_t _Vec>
		inline void step(_Mat& _Weights, _Mat&& _W_gradients, _Vec& _Biases, _Vec&& _B_gradients) {
			_W_gradients *= _Hp.learning_rate;
			_Weights -= _W_gradients;
			
			_B_gradients *= _Hp.learning_rate;
			_Biases -= _B_gradients;
		}

		inline friend std::ostream& operator<<(std::ostream& _Os, const sgd& _Sgd) {
			return _Os << "lr: " << _Sgd._Hp.learning_rate;
		}

	private:
		const hyperparameters<sgd> _Hp;
		const size_t _Layer_count;
	};
}