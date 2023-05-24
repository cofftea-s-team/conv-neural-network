#pragma once

#include "config.hpp"
#include "../base/utils.hpp"
namespace cnn {
	
	class linear {
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;
		using vector = typename config::vector;
		using dual_matrix = typename config::dual_matrix;
		
		template <class... _TLayers>
		friend class neural_network;
		friend class file;

		inline linear(size_t _InputSize, size_t _OutputSize)
			: _Weights(_InputSize, _OutputSize), _Bias(base::shape(1, _OutputSize))
		{
			host::matrix<value_type> _W_tmp(_InputSize, _OutputSize);
			host::vector<value_type, true> _B_tmp(base::shape(1, _OutputSize));
			utils::generate_normal(_W_tmp);
			utils::generate_normal(_B_tmp);
			_Weights = std::move(_W_tmp);
			_Bias = std::move(_B_tmp);
		}

		inline auto operator()(matrix& _Input) {
			if (_Weights.get_result() == nullptr) 
				_Weights.alloc_result(utils::get_mul_shape(_Input, _Weights));
			return _Input.mul_add_bias(_Weights, _Bias, _Weights.get_result());
		}

		template <class _Optimizer>
		inline auto backward(matrix& _Error, matrix& _Input, _Optimizer& _Opt) {
			_Opt.step(_Weights, _Input.T().mul(_Error), _Bias, _Error.sum1());
			return _Error.mul(_Weights.T());
		}
		
		dual_matrix _Weights;
		vector _Bias;
	};
}