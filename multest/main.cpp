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
//#include "python.hpp"

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
	std::cout << std::setprecision(15) << std::fixed;
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


// current implementation limits cuda::matrix to 2048*2048 elements

#define TIMER_START(x) auto _s##x = high_resolution_clock::now(); auto _1s##x = clock();
#define TIMER_END(x) auto _s2##x = high_resolution_clock::now(); auto _1s2##x = clock();
#define TIMER_RESULT(x, pre) auto count1##x = duration_cast<milliseconds>(_s2##x - _s##x).count(); auto count2##x = (_1s2##x - _1s##x); cout << '[' << ##pre << "]\n" << "user time: " << count1##x << "ms\nsys time: " << count2##x << "ms\n" << endl;

template<class _Ty>
_Ty mse_loss(cuda::matrix <_Ty> _Output, cuda::matrix <_Ty> _Target) {
	_Ty loss = 0;

	auto _Error = _Output - _Target;
	_Error *= _Error;
	auto _ErrorSumMat = _Error.to_host();
	
	for (auto&& e : _ErrorSumMat) {
		loss += e;
	}

	return loss / _Output.size();
}

struct proxy {
	// linear<_Ty>
};

template<class _Ty>
class linear {
	public:
		linear(int _InputSize, int _OutputSize) {
			w = cuda::matrix<_Ty>(_InputSize, _OutputSize);
			b = cuda::vector<_Ty>(_OutputSize);
			utils::generate_normal(w);
			utils::generate_normal(b);
	}
		auto operator()(cuda::matrix<_Ty>& _Input) {
			last_input = _Input;
			return _Input.mul(w);// +b;
	}
		auto backward(cuda::matrix<_Ty>& _Error) {
			auto _ErrorGrad = _Error.mul(transposed(w));
			auto _WGrad = transposed(last_input).mul(_Error);
			auto _BGrad = host::vector<_Ty>(b.size());
			auto _ErrorHost = _Error.to_host();
			for (int i = 0; i < _Error.size(); ++i) {
				_BGrad.data()[i % b.size()] += _ErrorHost.data()[i];
			}
			cuda::vector<_Ty> _BGradCuda = _BGrad;
			w -= _WGrad * 0.1;
			//cuda::matrix_mul_scalar(_BGradCuda, _BGradCuda, 0.01f);
			//cuda::matrix_sub(b, _BGradCuda, b);
			return _ErrorGrad;
		}

private:
	cuda::matrix<_Ty> w;
	cuda::vector<_Ty> b;
	cuda::matrix<_Ty> last_input;
};

template<class _Ty>
class nn {
public:
	using matrix = cuda::matrix<_Ty>;
	using vector = cuda::vector<_Ty>;
	nn() {
		layers.emplace_back(2, 4);
		layers.emplace_back(4, 1);
	}
	void train(int _Epochs, matrix _Input, matrix _Target) {
		for (int i = 0; i < _Epochs; ++i) {
			auto output = _Input;
			for (auto&& layer : layers) {
				output = layer(output);
				output.activate<sigmoid>();
			}
			auto loss = mse_loss(output, _Target);
			if (i % 1000 == 0)
				cout << "loss: " << loss << endl;
			auto error = output - _Target;
			for (auto&& layer : std::views::reverse(layers)) {
				error = layer.backward(error);
			}
			
		}
	}
	auto predict(matrix _Input) {
		auto output = _Input;
		for (auto&& layer : layers) {
			output = layer(output);
			output.activate<sigmoid>();
		}
		return output;
	}

private:
	std::vector<linear<_Ty>> layers;
};


int _main(std::span<std::string_view> args) {
	nn<float> model;
	host::matrix<float> input(8, 2);
	host::matrix<float> output(8, 1);

	// xor
	for (int i = 0; i < 8; ++i) {
		input.data()[i * 2] = i & 1;
		input.data()[i * 2 + 1] = (i >> 1) & 1;
		output.data()[i] = (i & 1) ^ ((i >> 1) & 1);
	}
	
	cout << input << endl;
	cout << output << endl;

	model.train(10000, input, output);

	// create a new input
	input = host::matrix<float>(4, 2);
	for (int i = 0; i < 4; ++i) {
		input.data()[i * 2 + 1] = 0;
		input.data()[i * 2] = (i >> 1) & 1;
	}
	cout << input << endl;

	cout << model.predict(input) << endl;

	return 0;
}

