#pragma once
#include "config.hpp"
#include "activations.hpp"

namespace cnn {
	
	struct mse {
		using value_type = config::value_type;
		
		template <matrix_t _TMatrix>
		static auto loss(const _TMatrix& _Output, const _TMatrix& _Target) {
			value_type loss = 0.f;

			auto _Error = _Output - _Target;
			_Error *= _Error;

			for (size_t i = 0; i < _Error.size(); ++i) {
				if constexpr (_TMatrix::alloc::is_cuda())
					loss += cuda::from_cuda(&_Error.data()[i]);
				else
					loss += _Error.data()[i];
			}

			return loss / _Error.size();
		}

		template <matrix_t _TMatrix>
		static auto derivative(const _TMatrix& _Output, const _TMatrix& _Target) {
			return _Output - _Target;
		}
	};

	struct cross_entropy {
		using value_type = config::value_type;
		
		template <matrix_t _TMatrix>
		static auto loss(const _TMatrix& _Output, const _TMatrix& _Target) {
			value_type loss = 0.f;
			
			for (size_t i = 0; i < _Output.size(); ++i) {
				if constexpr (_TMatrix::alloc::is_cuda())
					loss += cuda::from_cuda(&_Target.data()[i]) * std::log(cuda::from_cuda(&_Output.data()[i]) + 1e-8f);
				else
					loss += _Target.data()[i] * std::log(_Output.data()[i] + 1e-8f);
			}

			return -loss / _Output.size();
		}

		template <matrix_t _TMatrix>
		static auto derivative(const _TMatrix& _Output, const _TMatrix& _Target) {
			auto _Diff = _Output - _Target;
			_Diff /= _Target.cols();
			return _Diff;
		}
	};
}