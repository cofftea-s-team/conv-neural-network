#pragma once

#include "config.hpp"
#include "optimizers/adam.hpp"
#include "optimizers/sgd.hpp"

namespace cnn {
	using namespace optimizers;
	
	template <class _Optimizer>
	concept optimizer = requires (_Optimizer _Opt) {
		_Opt.step;
		std::cout << _Opt;
	};
}