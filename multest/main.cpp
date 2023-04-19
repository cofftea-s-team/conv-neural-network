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
	for (int i = 0; i < _Output.size(); ++i) {
		loss += (_Output.data()[i] - _Target.data()[i]) * (_Output.data()[i] - _Target.data()[i]);
	}
	return loss / _Output.size();
}



template<class _Ty>
class nn {
public:
	using matrix = cuda::matrix<_Ty>;
	using vector = cuda::vector<_Ty>;
	nn(size_t _In, size_t _Hid, size_t _Out) {
		w1 = matrix(_In, _Hid);
		utils::generate_normal(w1);
		w2 = matrix(_Hid, _Out);
		utils::generate_normal(w2);
	}
	void train(int _Epochs, matrix _Input, matrix _Output) {
		host::matrix<_Ty> lrm(_Output.shape());
		for (int i = 0; i < lrm.size(); ++i) {
			lrm.data()[i] = lr;
		}
		auto lrm2 = lrm.to_cuda();
		for (int i = 0; i < _Epochs; ++i) {
			auto h1 = _Input.mul(w1);
			h1.activate<sigmoid>();
			auto h2 = h1.mul(w2);
			
			auto e = _Output - h2;
			e *= lrm2;
			/*auto loss = mse_loss(h2, _Output);
			if (i % 1000 == 0) {
				cout << "loss: " << loss << endl;
			}*/
			matrix h1t = h1.T();
			w2 += h1t.mul(e);
			auto dhr = e.mul(transposed(w2));
			matrix dh = h1;
			dh.activate<bsigmoid>();
			dhr *= dh;
			matrix dwt = _Input.T();
			w1 += dwt.mul(dhr);
		}
	}
	auto predict(matrix _Input) {
		auto h1 = _Input.mul(w1);
		h1.activate<sigmoid>();
		auto h2 = h1.mul(w2);
		return h2;
	}

private:
	_Ty lr = 0.1;
	matrix w1 { 1, 1 }, w2 {1, 1};
};

int _main(std::span<std::string_view> args) {
	nn<float> nn(2, 3, 1);
	host::matrix<float> input(8, 2);
	host::matrix<float> output(8, 1);

	// xor
	for (int i = 0; i < 4; ++i) {
		input.data()[i * 2] = i & 1;
		input.data()[i * 2 + 1] = (i >> 1) & 1;
		output.data()[i] = (i & 1) ^ ((i >> 1) & 1);
	}
	
	cout << input << endl;
	cout << output << endl;

	nn.train(1000, input, output);

	// create a new input
	input = host::matrix<float>(4, 2);
	for (int i = 0; i < 4; ++i) {
		input.data()[i * 2 + 1] = 0;
		input.data()[i * 2] = (i >> 1) & 1;
	}
	cout << input << endl;

	cout << nn.predict(input) << endl;

	return 0;
}

