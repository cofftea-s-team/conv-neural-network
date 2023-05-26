#pragma once
#include <iostream>
#include "../host/matrix.hpp"
#include "../host/dual_matrix.hpp"
#include "../host/vector.hpp"

#include "../cuda/matrix.hpp"
#include "../cuda/dual_matrix.hpp"
#include "../cuda/vector.hpp"

#include <algorithm>
#include <ranges>
#include <execution>

namespace cnn {
	struct linear;

	enum class device_type {
		cpu,
		cuda
	};
	
	struct config {
		using value_type = float;
		static constexpr device_type device = device_type::cuda;
		
		using matrix = std::conditional_t<device == device_type::cpu, host::matrix<value_type>, cuda::matrix<value_type>>;
		using vector = std::conditional_t<device == device_type::cpu, host::vector<value_type, true>, cuda::vector<value_type, true>>;
		using dual_matrix = std::conditional_t<device == device_type::cpu, host::dual_matrix<value_type>, cuda::dual_matrix<value_type>>;
	};

	template <class _Ty>
	concept cuda_matrix = std::is_same_v<_Ty, cuda::matrix<config::value_type>> ||
		std::is_same_v<_Ty, cuda::dual_matrix<config::value_type>> ||
		std::is_same_v<_Ty, cuda::vector<config::value_type, true>>;

	template <class _Ty>
	concept host_matrix = std::is_same_v<_Ty, host::matrix<config::value_type>> ||
		std::is_same_v<_Ty, host::dual_matrix<config::value_type>> ||
		std::is_same_v<_Ty, host::vector<config::value_type, true>>;

	template <class _Ty, class... _TArgs>
	concept matrix_t = cuda_matrix<_Ty> || host_matrix<_Ty>;

	template <class _Ty>
	concept loss_fn = requires () {
		_Ty::loss;
		_Ty::derivative;
	};

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

	inline size_t _Argmax_helper(const config::value_type* _Data, size_t N) {
		size_t index = 0;
		auto max = _Data[0];
		for (size_t i = 1; i < N; ++i) {
			if (_Data[i] > max) {
				max = _Data[i];
				index = i;
			}
		}
		return index;
	}

	inline std::vector<size_t> argmax(const host::matrix<config::value_type>& _Output) {
		size_t N = _Output.cols();
		size_t M = _Output.rows();
		auto* _Data = _Output.data();
		
		std::vector<size_t> _Res(M);
		for (size_t i = 0; i < M; ++i) {
			_Res[i] = _Argmax_helper(&_Data[i * N], N);
		}

		return _Res;
	}

	template <matrix_t _TMatrix>
	inline auto accuracy(const _TMatrix& _Preds, const _TMatrix& _Labels) {
		using value_type = config::value_type;
		
		host::matrix<value_type> _Res = _Preds * _Labels;
		value_type _Sum = 0;
		for (size_t i = 0; i < _Res.size(); ++i) {
			_Sum += round(_Res.data()[i]);
		}
		return _Sum / _Preds.rows();
	}
}