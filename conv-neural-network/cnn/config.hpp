#pragma once
#include <iostream>
#include "../host/matrix.hpp"
#include "../host/dual_matrix.hpp"
#include "../host/vector.hpp"

#include "../cuda/matrix.hpp"
#include "../cuda/dual_matrix.hpp"
#include "../cuda/vector.hpp"


namespace cnn {
	struct linear;

	enum class device_type {
		cpu,
		cuda
	};
	
	struct config {
		using value_type = float;
		static constexpr device_type device = device_type::cpu;
		
		using matrix = std::conditional_t<device == device_type::cpu, host::matrix<value_type>, cuda::matrix<value_type>>;
		using vector = std::conditional_t<device == device_type::cpu, host::vector<value_type, true>, cuda::vector<value_type, true>>;
		using dual_matrix = std::conditional_t<device == device_type::cpu, host::dual_matrix<value_type>, cuda::dual_matrix<value_type>>;
	};

	template <class _Ty>
	concept matrix_t = std::is_same_v<_Ty, config::matrix> || 
		std::is_same_v<_Ty, config::dual_matrix> || std::is_same_v<_Ty, config::vector>;

	template <class... _TLayers>
	struct _Count_linears;

	template <>
	struct _Count_linears<> {
		static constexpr size_t value = 0;
	};

	template <class _TLayer, class... _TLayers>
	struct _Count_linears<_TLayer, _TLayers...> {
		static constexpr size_t value = _Count_linears<_TLayers...>::value + std::is_same_v<_TLayer, cnn::linear>;
	};
	
	namespace optimizers {
		
		template<class _Optimizer>
		struct hyperparameters;
	}

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

	inline auto crossentropy_loss(const typename config::matrix& _Output, const typename config::matrix& _Target) {
		using value_type = typename config::value_type;

		
	}
}