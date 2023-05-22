#pragma once

#include "config.hpp"
#include "optimizers/adam.hpp"
#include "optimizers/sgd.hpp"

namespace cnn {
	using adam = optimizers::adam;
	using sgd = optimizers::sgd;

	template <class _Optimizer>
	using hyperparameters = optimizers::hyperparameters<_Optimizer>;

	template <class _Optimizer>
	concept optimizer = requires (_Optimizer _Opt) {
		_Opt.step;
		std::cout << _Opt;
	};
}