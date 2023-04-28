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
			utils::generate_normal(_Weights);
			utils::generate_normal(_Bias);
		}

		inline auto operator()(matrix& _Input) {
			return _Input.mul(_Weights) + _Bias;
		}

		matrix _Weights;
		vector _Bias;
	};
}