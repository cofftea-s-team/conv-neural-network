#pragma once
#include "config.hpp"

namespace cnn {

	class dropout {
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;
		using vector = typename config::vector;
		
		inline dropout(value_type _Rate)
			: _Dropout_rate(_Rate) 
		{ }

		inline void operator()(matrix& _Input) const {
			_Input.dropout(_Dropout_rate);
		}
	private:
		const value_type _Dropout_rate;
	};
}