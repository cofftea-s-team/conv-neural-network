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
	host::algebra::details::indices;

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

int _main(std::span<std::string_view> args) {

	host::matrix<float> m(16, 8);
	utils::generate_uniform(m);

	cout << m << endl;

	auto t1 = m.T();

	cout << t1 << endl;

	auto t2 = t1.T();

	cout << t2 << endl;

	return 0;
}

