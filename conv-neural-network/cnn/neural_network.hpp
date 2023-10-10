#pragma once
#include "logger.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include "config.hpp"
#include "optimizers.hpp"
#include "loss.hpp"
#include <vector>
#include <tuple>

namespace cnn {

	enum class _Lt { _None, _Linear, _Activation, _Dropout };

	template <class _TLayer>
	consteval _Lt _Select_layer_type() {
		if constexpr (std::is_same_v<_TLayer, linear>)
			return _Lt::_Linear;
		else if constexpr (requires(_TLayer, float x) { _TLayer::forward(x); })
			return _Lt::_Activation;
		else if constexpr (requires(_TLayer, float x) { _TLayer::forward(x, x); })
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

		static constexpr size_t linear_count = _Count_linears<_TLayers...>::value;

		friend class file;

		constexpr neural_network(_TLayers&&... _Sequential) noexcept
			: _Layers(std::forward<_TLayers>(_Sequential)...)
		{ }

		template <class _TLoss, class _TOptimizer>
		inline void train(size_t _Epochs, matrix& _Input, matrix& _Target, _TOptimizer& _Opt, logger& _Logger) {
			for (size_t i = 0; i < _Epochs; ++i) {
				_Train_once<_TLoss>(_Input, _Target, _Opt);
				matrix _Output = predict(_Input);

				//_Fn(i, loss(_Output, _Target), _Output, _Opt);
				_Logger.log<_TLoss>(i, _Output, _Target);
			}
		}

		template <bool _Train = false>
		inline auto predict(matrix& _Input) {
			_Outputs.clear();
			_Outputs.emplace_back(_Input);
			// forward pass
			utils::for_each(_Layers, [&]<class _TLayer>(_TLayer & _Layer) {
				static constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Forward_linear(_Layer);
				else if constexpr (_Sel == _Lt::_Activation)
					_Forward_activation<_TLayer>();
				else if constexpr (_Sel == _Lt::_Dropout)
					_Forward_dropout<_Train>(_Layer);
				else
					static_assert(std::_Always_false<_TLayer>, "Invalid layer type");
				//std::cout << "out: " << _Outputs.back() << std::endl;
			});

			return _Outputs.back();
		}

	private:
		template <class _TLoss, class _TOptimizer>
		inline void _Train_once(matrix& _Input, matrix& _Target, _TOptimizer& _Opt) {
			predict<true>(_Input);
			auto _Error = _TLoss::derivative(_Outputs.back(), _Target);

			_Outputs.pop_back();
			_Outputs.emplace_back(std::move(_Error));

			utils::rfor_each(_Layers, [&]<class _TLayer>(_TLayer & _Layer) {
				static constexpr auto _Sel = _Select_layer_type<_TLayer>();

				if constexpr (_Sel == _Lt::_Linear)
					_Backward_linear(_Layer, _Opt);
				else if constexpr (_Sel == _Lt::_Activation)
					_Backward_activation<_TLayer>();
				else if constexpr (_Sel == _Lt::_Dropout)
					_Backward_dropout<true>(_Layer);
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
			matrix _Input = _Outputs.back();
			_Input.activate<_Act_fn>();
			_Outputs.emplace_back(std::move(_Input));
		}

		template <class _Optimizer>
		inline void _Backward_linear(linear& _Layer, _Optimizer& _Opt) {
			matrix _Error = std::move(_Outputs.back());
			_Outputs.pop_back();
			matrix _Input = std::move(_Outputs.back());
			_Outputs.pop_back();
			auto _Res = _Layer.backward(_Error, _Input, _Opt);
			_Outputs.emplace_back(std::move(_Res));
		}

		template <activation_fn_t _Act_fn>
		inline void _Backward_activation() {
			matrix _Error = std::move(_Outputs.back());
			_Outputs.pop_back();
			matrix _Activated = std::move(_Outputs.back());
			//std::cout << _Activated << std::endl;
			_Outputs.pop_back();
			_Activated.backward<_Act_fn>();
			//std::cout << _Activated << std::endl;
			_Error *= _Activated;
			_Outputs.emplace_back(std::move(_Error));
		}

		template <bool _Train>
		inline void _Forward_dropout(const dropout& _Layer) {
			if constexpr (_Train)
				_Layer(_Outputs.back());
		}

		template <bool _Train>
		inline void _Backward_dropout(const dropout& _Layer) {
			if constexpr (_Train)
				_Layer.backward(_Outputs.back());
		}

		std::tuple<_TLayers...> _Layers;
		std::vector<matrix> _Outputs;
	};
}