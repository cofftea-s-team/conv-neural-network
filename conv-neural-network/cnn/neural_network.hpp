#pragma once
#include "linear.hpp"
#include "dropout.hpp"
#include "config.hpp"
#include <vector>
#include <tuple>

namespace cnn {

	enum class _Lt { _None, _Linear, _Activation, _Dropout };
	
	template <class _TLayer>
	consteval _Lt _Select_layer_type() {
		if constexpr (std::is_same_v<_TLayer, linear>)
			return _Lt::_Linear;
		else if constexpr (requires(_TLayer) { _TLayer::forward; })
			return _Lt::_Activation;
		else if constexpr (std::is_same_v<_TLayer, dropout>)
			return _Lt::_Dropout;
		else
			return _Lt::_None;
	}

	template <class... _TLayers>
	class neural_network
	{
	public:
		using value_type = typename config::value_type;
		using matrix = typename config::matrix;
		using vector = typename config::vector;
		using dual_matrix = typename config::dual_matrix;

		value_type learning_rate = 0.00001f;

		friend class file;

		constexpr neural_network(_TLayers&&... _Sequential) noexcept
			: _Layers(std::forward<_TLayers>(_Sequential)...)
		{ }

		template <class _Lambda>
		inline void train(size_t _Epochs, matrix& _Input, matrix& _Target, _Lambda _Fn) {
			for (size_t i = 0; i < _Epochs; ++i) {
				_Train_once(_Input, _Target);
				matrix _Output = predict(_Input);

				if (i % 1000 == 0) {
					_Fn(i, loss(_Output, _Target), _Output );
					//learning_rate *= 1.1;
				}
			}
		}

		template <bool _Train = false>
		inline auto predict(matrix& _Input) {
			_Outputs.clear();
			_Outputs.emplace_back(_Input);
			// forward pass
			utils::for_each(_Layers, [&]<class _TLayer>(_TLayer & _Layer) {
				constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Forward_linear(_Layer);
				else if constexpr (_Sel == _Lt::_Activation)
					_Forward_activation<_TLayer>();
				else if constexpr (_Sel == _Lt::_Dropout) {
					if constexpr (_Train) _Dropout(_Layer);
				}
				else
					static_assert(std::_Always_false<_TLayer>, "Invalid layer type");
			});

			return _Outputs.back();
		}

	private:
		inline void _Train_once(matrix& _Input, matrix& _Target) {
			predict<true>(_Input);

			auto _Error = (_Target - _Outputs.back()) * learning_rate;
			_Outputs.pop_back();
			_Outputs.emplace_back(std::move(_Error));

			utils::rfor_each<sizeof...(_TLayers)>(_Layers, [&]<class _TLayer>(_TLayer & _Layer) {
				constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Backward_linear(_Layer);
				else if constexpr (_Sel == _Lt::_Activation)
					_Backward_activation<_TLayer>();
				else if constexpr (_Sel == _Lt::_Dropout)
				{ }
				else
					static_assert(std::_Always_false<_TLayer>, "Invalid layer type");

			});

		}

		inline void _Forward_linear(linear& _Layer) {
			matrix& _Input = _Outputs.back();
			matrix _Result = _Layer(_Input);
			_Outputs.emplace_back(std::move(_Result));
		}

		template <activation_fn_t _Act_fn>
		inline void _Forward_activation() {
			matrix& _Input = _Outputs.back();
			_Input.activate<_Act_fn>();
		}

		inline void _Backward_linear(linear& _Layer) {
			matrix _Error = std::move(_Outputs.back());
			_Outputs.pop_back();
			matrix& _Input = _Outputs.back();
			matrix _Delta = _Input.T().mul(_Error);
			auto _BiasDelta = _Error.sum1();
			_Layer._Weights += _Delta;
			_Layer._Bias += _BiasDelta;

			_Prev_weights = &_Layer._Weights;
			_Outputs.push_back(std::move(_Error));
		}

		template <activation_fn_t _Act_fn>
		inline void _Backward_activation() {
			matrix _Error = std::move(_Outputs.back());
			_Outputs.pop_back();
			matrix _Input = std::move(_Outputs.back());
			_Outputs.pop_back();
			_Input.backward<_Act_fn>();

			_Outputs.emplace_back(_Error.mul(_Prev_weights->T()) * _Input);
		}

		inline void _Dropout(const dropout& _Layer) {
			_Layer(_Outputs.back());
		}

		std::tuple<_TLayers...> _Layers;
		std::vector<matrix> _Outputs;
		matrix* _Prev_weights = nullptr;
	};
}