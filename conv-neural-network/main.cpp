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
	std::cout << std::setprecision(5) << std::fixed;
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

	using value_type = typename settings::value_type;

	host::matrix<value_type> _Error = _Output - _Target;
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
		utils::generate_normal(_Weights);
		utils::generate_normal(_Bias);
	}
	
	inline auto operator()(matrix& _Input) {
		return _Input.mul(_Weights) + _Bias;
	}

	matrix _Weights;
	host::vector<value_type, true> _Bias;
};


class dropout {
public:
	using value_type = typename settings::value_type;
	using matrix = typename settings::matrix;
	using vector = typename settings::vector;

	inline dropout(value_type _Val) 
		: _Probability(_Val)
	{ }

	inline void operator()(matrix& _Input) {
		
	}
private:


	const value_type _Probability;
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
	
	value_type learning_rate = 0.00001f;

	constexpr neural_network(_TLayers&&... _Sequential) noexcept
		: _Layers(std::forward<_TLayers>(_Sequential)...)
	{ }

	inline void train(size_t _Epochs, matrix& _Input, matrix& _Target) {
		for (size_t i = 0; i < _Epochs; ++i) {
			_Train_once(_Input, _Target);
			matrix _Output = predict(_Input);
			
			if (i % 1000 == 0) {
				cout << "Epoch: " << i << " Loss: " << loss(_Output, _Target) << endl;
				learning_rate *= 1.1;
			}
		}
	}

	inline auto predict(matrix& _Input) {
		_Outputs.clear();
		_Outputs.emplace_back(_Input);
		// forward pass
		utils::for_each(_Layers, [&]<class _TLayer>(_TLayer& _Layer) {
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

	std::tuple<_TLayers...> _Layers;
	std::vector<matrix> _Outputs;
	matrix* _Prev_weights = nullptr;
};

#include <fstream>
// driver code



void normalize_input(host::matrix<float>& _Input, float _MinX, float _MaxX, float _MinY, float _MaxY) {
	for (size_t i = 0; i < _Input.rows(); ++i) {
		_Input(i, 0) = (_Input(i, 0) - _MinX) / (_MaxX - _MinX) - 0.5f;
		_Input(i, 1) = (_Input(i, 1) - _MinY) / (_MaxY - _MinY) - 0.5f;
	}
}

void denormalize(host::matrix<float>& _Input, float _MinX, float _MaxX, float _MinY, float _MaxY) {
	for (size_t i = 0; i < _Input.rows(); ++i) {
		_Input(i, 0) = (_Input(i, 0) + 0.5f) * (_MaxX - _MinX) + _MinX;
		_Input(i, 1) = (_Input(i, 1) + 0.5f) * (_MaxY - _MinY) + _MinY;
	}
}

int _main(std::span<std::string_view> args) {
	using value_type = typename settings::value_type;
	using matrix = typename settings::matrix;
	using vector = typename settings::vector;
	
	std::ifstream file("input.txt");
	
	matrix input(100, 2);

	while (!file.eof()) {
		for (size_t i = 0; i < input.rows(); ++i) {
			file >> input(i, 0) >> input(i, 1);
		}
	}
	normalize_input(input, -1.5, 2.5, -1, 2);

	file.close();
	
	std::ifstream file2("labels.txt");

	matrix target(100, 1);

	while (!file2.eof()) {
		for (size_t i = 0; i < target.rows(); ++i) {
			file2 >> target(i, 0);
		}
	}

	neural_network model(
		linear(2, 32),
		relu(),
		linear(32, 24),
		relu(),
		linear(24, 1)
	);
	
	model.train(8000, input, target);

	
	std::ifstream file3("grid.txt");
	
	matrix grid(100000, 2);

	while (!file3.eof()) {
		for (size_t i = 0; i < grid.rows(); ++i) {
			file3 >> grid(i, 0) >> grid(i, 1);
		}
	}

	normalize_input(grid, -1.5, 2.5, -1, 2);

	auto res = model.predict(grid);

	std::ofstream file4("output.txt");
	
	for (size_t i = 0; i < res.rows(); ++i) {
		file4 << res(i, 0) << '\n';
	}
	
	return 0;
}

// -1.5, 2.5
// -1, 2