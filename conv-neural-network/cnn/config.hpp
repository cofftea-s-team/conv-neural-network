#pragma once
#include "../host/matrix.hpp"
#include "../host/dual_matrix.hpp"
#include "../host/vector.hpp"

#include "../cuda/matrix.hpp"
#include "../cuda/dual_matrix.hpp"
#include "../cuda/vector.hpp"


namespace cnn {

	struct config {
		using value_type = float;
		using matrix = cuda::matrix<value_type>;
		using vector = cuda::vector<value_type, true>;
		using dual_matrix = cuda::dual_matrix<value_type>;
	};

	inline auto loss(const typename config::matrix& _Output, const typename config::matrix& _Target) {
		typename config::value_type loss = 0.f;

		using value_type = typename config::value_type;

		host::matrix<value_type> _Error = _Output - _Target;
		_Error *= _Error;

		for (auto&& e : _Error) {
			loss += e;
		}

		return loss / _Output.size();
	}
}