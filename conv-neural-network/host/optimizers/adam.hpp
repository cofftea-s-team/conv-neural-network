#pragma once
#include <vector>
#include "../matrix.hpp"
#include "../../cnn/config.hpp"
namespace host::optimizers
{
	class adam
	{
		using value_type = typename cnn::config::value_type;
	public:
		adam(double lr = 1e-3, double decay = 1e-4, double epsilon = 1e-7, double beta1 = 0.9, double beta2 = 0.999, size_t layer_count = 0) {
			_Lr = lr;
			_Decay = decay;
			_Epsilon = epsilon;
			_Beta1 = beta1;
			_Beta2 = beta2;
			_Iters = 0;
			_Layer_count = layer_count;
			_Current_lr = _Lr * (1.0 / (1.0 + _Decay * (2.0 * _Iters / _Layer_count)));
		}

		void step(matrix<value_type>& _Tensor, matrix<value_type>& _Gradient) {
			if (_Tensor_cache.size() < _Layer_count) {
				_Tensor_cache.push_back(matrix<value_type>(_Tensor.shape()));
				_Tensor_momentums.push_back(matrix<value_type>(_Tensor.shape()));
			}
			_Tensor_cache[_Layer_count - 1] = _Tensor_cache[_Layer_count - 1] * _Beta1 + _Gradient * (1 - _Beta1);
			_Tensor_momentums[_Layer_count - 1] = _Tensor_momentums[_Layer_count - 1] * _Beta2 + _Gradient * _Gradient * (1 - _Beta2);
			_Current_lr = _Lr * std::sqrt(1 - std::pow(_Beta2, _Iters)) / (1 - std::pow(_Beta1, _Iters));
			_Tensor = _Tensor - _Tensor_cache[_Layer_count - 1] / (_Tensor_momentums[_Layer_count - 1] + _Epsilon) * _Current_lr;
			_Iters++;
		}

	private:
		size_t _Iters;
		double _Current_lr;
		double _Lr;
		double _Decay;
		double _Epsilon;
		double _Beta1;
		double _Beta2;
		size_t _Layer_count;
		std::vector<matrix<value_type>> _Tensor_cache;
		std::vector<matrix<value_type>> _Tensor_momentums;
	};
}