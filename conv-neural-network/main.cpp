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

#pragma region better main impl
inline void preload() {
	cout << "Loading...";

	std::ios_base::sync_with_stdio(false);
	std::cout << std::setprecision(3) << std::fixed;
	cuda::matrix<float> f(16, 16);
	f.mul(transposed(f));
	host::algebra::parallel::details::indices;

	cout << " done!" << endl;
}

int main(int argc, const char* const argv[]) {
	[[nodiscard]] int _main([[maybe_unused]] std::span<std::string_view>);
	std::array<std::string_view, 255> args;

	auto enumerate = [&, id = 0]<class _Ty>(_Ty & _Val) mutable {
		return std::pair<uint16_t, _Ty&>(id++, _Val);
	};

	for (auto&& [i, val] : args | stdrv::transform(enumerate) | stdrv::take(argc)) {
		val = std::string_view(argv[i]);
	}

	preload();
	int exit_code = _main(args);

	return exit_code;
}
#pragma endregion
 
#define TIMER_START(x) auto _s##x = high_resolution_clock::now(); auto _1s##x = clock();
#define TIMER_END(x) auto _s2##x = high_resolution_clock::now(); auto _1s2##x = clock();
#define TIMER_RESULT(x, pre) auto count1##x = duration_cast<milliseconds>(_s2##x - _s##x).count(); auto count2##x = (_1s2##x - _1s##x); cout << '[' << ##pre << "]\n" << "user time: " << count1##x << "ms\nsys time: " << count2##x << "ms\n" << endl;

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

		utils::rfor_each<sizeof...(_TLayers)>(_Layers, [&]<class _TLayer>(_TLayer& _Layer) {
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

// driver code
int _main(std::span<std::string_view> args) {

	using value_type = typename settings::value_type;
	using matrix = typename settings::matrix;
	using vector = typename settings::vector;
	
	neural_network model = {
		linear(2, 3),
		sigmoid(),
		linear(3, 2),
		sigmoid(),
		linear(2, 1)
	};
	

	const size_t batch_size = 4;
	matrix input(batch_size, 2);
	matrix output(batch_size, 1);

	// xor
	for (int i = 0; i < batch_size; ++i) {
		input.data()[i * 2] = i & 1;
		input.data()[i * 2 + 1] = (i >> 1) & 1;
		output.data()[i] = (i & 1) ^ ((i >> 1) & 1);
	}

	cout << input << endl;
	cout << output << endl;

	model.train(5000, input, output);

	matrix test(4, 2);
	for (int i = 0; i < 4; ++i) {
		test.data()[i * 2 + 1] = 0;
		test.data()[i * 2] = (i >> 1) & 1;
	}
	cout << test << endl;

	cout << model.predict(test) << endl;

	
	return 0;
}