#pragma once

#include "config.hpp"

namespace cnn {
	
	class linear {
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;
		using vector = typename config::vector;

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
			_Weights = _W_tmp;
			_Bias = _B_tmp;
		}

		inline auto operator()(matrix& _Input) {
			return _Input.mul(_Weights) + _Bias;
		}

		matrix _Weights;
		vector _Bias;
	};
}