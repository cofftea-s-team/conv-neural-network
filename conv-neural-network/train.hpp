#define DEBUG
#include "host/matrix.hpp"
#include "host/vector.hpp"
#include "host/dual_matrix.hpp"

#include "cuda/matrix.hpp"
#include "cuda/vector.hpp"
#include "cuda/dual_matrix.hpp"

#include "vector_view.hpp"
#include <iostream>
#include <iomanip>

#include <chrono>
#include <random>
#include <ranges>
#include <span>
#include <algorithm>
#include "utils.hpp"
#include <tuple>

using std::cout;
using std::endl;

namespace stdr = std::ranges;
namespace stdrv = std::ranges::views;

using namespace std::chrono;
using namespace base;


struct settings {
	using value_type = float;
	using matrix = host::matrix<value_type>;
	using vector = host::vector<value_type>;
	using dual_matrix = host::dual_matrix<value_type>;
};

inline auto loss(const typename settings::matrix& _Output, const typename settings::matrix& _Target) {
	typename settings::value_type loss = 0.f;

	auto _Error = _Output - _Target;
	_Error *= _Error;

	for (auto&& e : _Error) {
		loss += e;
	}

	return loss / _Output.size();
}

class linear {
public:
	using value_type = typename settings::value_type;
	using matrix = typename settings::matrix;
	using vector = typename settings::vector;

	template <class... _TLayers>
	friend class neural_network;

	inline linear(size_t _InputSize, size_t _OutputSize)
		: _Weights(_InputSize, _OutputSize), _Bias(_OutputSize)
	{
		utils::generate_uniform(_Weights);
		utils::generate_uniform(_Bias);
	}

	inline auto operator()(matrix& _Input) {
		return _Input.mul(_Weights); // + _Bias;
	}

	matrix _Weights;
	vector _Bias;
};


enum class _Lt { _None, _Linear, _Activation };

template <class _TLayer>
consteval _Lt _Select_layer_type() {
	if constexpr (std::is_same_v<_TLayer, linear>)
		return _Lt::_Linear;
	if constexpr (requires(_TLayer) { _TLayer::forward; })
		return _Lt::_Activation;
	else
		return _Lt::_None;
}


template <class... _TLayers>
class neural_network
{
public:
	using value_type = typename settings::value_type;
	using matrix = typename settings::matrix;
	using vector = typename settings::vector;
	using dual_matrix = typename settings::dual_matrix;

	static constexpr value_type learning_rate = 0.1f;

	constexpr neural_network(_TLayers&&... _Sequential) noexcept
		: _Layers(std::forward<_TLayers>(_Sequential)...)
	{ }

	/*
	constexpr neural_network(_TLayers&&... _Sequential) noexcept
		: _Layers(std::forward<_TLayers>(_Sequential)...)
	{ }
	make a void function that takes initializer list
	*/


	inline void train(size_t _Epochs, matrix& _Input, matrix& _Target) {
		for (size_t i = 0; i < _Epochs; ++i) {
			_Train_once(_Input, _Target);
			matrix _Output = predict(_Input);

			if (i % 500 == 0)
				cout << "Epoch: " << i << " Loss: " << loss(_Output, _Target) << endl;
		}
	}

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
			else
				static_assert(std::_Always_false<_TLayer>, "Invalid layer type");
		});

		return _Outputs.back();
	}

private:
	inline void _Train_once(matrix& _Input, matrix& _Target) {
		predict(_Input);

		auto _Error = (_Target - _Outputs.back()) * learning_rate;
		_Outputs.pop_back();
		_Outputs.emplace_back(std::move(_Error));

		utils::rfor_each<sizeof...(_TLayers)>(_Layers, [&]<class _TLayer>(_TLayer & _Layer) {
			constexpr auto _Sel = _Select_layer_type<_TLayer>();

			if constexpr (_Sel == _Lt::_Linear)
				_Backward_linear(_Layer);
			else if constexpr (_Sel == _Lt::_Activation)
				_Backward_activation<_TLayer>();
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
		_Layer._Weights += _Delta;

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

	std::tuple<_TLayers...> _Layers;
	std::vector<matrix> _Outputs;
	matrix* _Prev_weights = nullptr;
};

neural_network model = {
	linear(2, 4),
	relu(),
	linear(4, 1)
};

class py_nn {
	using value_type = typename settings::value_type;
	using matrix = typename settings::matrix;
	using vector = typename settings::vector;
	
public:
	inline void train(size_t _Epochs, matrix& _Input, matrix& _Target) {
		model.train(_Epochs, _Input, _Target);
	}
	inline auto predict(matrix& _Input) {
		return model.predict(_Input);
	}
};