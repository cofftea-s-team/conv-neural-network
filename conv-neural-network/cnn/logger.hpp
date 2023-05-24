#pragma once
#include "config.hpp"
#include <functional>
namespace cnn {
	
	class logger {
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;

		inline logger(size_t _Diff, const std::function<void(size_t, value_type, value_type)>& _Log_fn) 
			: _Diff(_Diff), _Fn(_Log_fn) 
		{ }

		template <loss_fn _TLoss>
		inline void log(size_t _It, const matrix& _Preds, const matrix& _Target) {
			if (!(_It % _Diff == 0)) return;
			_Fn(_It, _TLoss::loss(_Preds, _Target), accuracy(_Preds, _Target));
		}

	private:
		size_t _Diff;
		std::function<void(size_t, value_type, value_type)> _Fn;
	};
}