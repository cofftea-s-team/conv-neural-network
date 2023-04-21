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

#include <functional>
// current implementation limits cuda::matrix to 2048*2048 elements

#define TIMER_START(x) auto _s##x = high_resolution_clock::now(); auto _1s##x = clock();
#define TIMER_END(x) auto _s2##x = high_resolution_clock::now(); auto _1s2##x = clock();
#define TIMER_RESULT(x, pre) auto count1##x = duration_cast<milliseconds>(_s2##x - _s##x).count(); auto count2##x = (_1s2##x - _1s##x); cout << '[' << ##pre << "]\n" << "user time: " << count1##x << "ms\nsys time: " << count2##x << "ms\n" << endl;

int _main(std::span<std::string_view> args) {

	host::matrix<float> m(8, 4);
	host::matrix<float> m2(4, 8);
	host::vector<float> v(8);

	auto enumerate = [&, id = 0]<class _Ty>(_Ty & _Val) mutable { return std::pair<size_t, _Ty&>(id++, _Val); };
	for (auto&& [i, val] : m | stdrv::transform(enumerate)) {
		val = i % m.cols() + i / m.cols();
	}
	for (auto&& [i, val] : v | stdrv::transform(enumerate)) {
		val = i % m.cols();
	}
	
	using matrix = cuda::matrix<float>;
	using vector = cuda::vector<float>;;

	for (auto&& [i, val] : m2 | stdrv::transform(enumerate)) {
		val = 1;
		if (i + 1 == m2.cols()) break;
	}
	cuda::matrix<float> a(m2);
	
	auto b = m.to_cuda();
	cuda::vector<float> v1(v);

	cout << v1 << endl;
	cout << b << endl;
	v1 += b.sum0();
	cout << v1 << endl;

	return 0;
}

