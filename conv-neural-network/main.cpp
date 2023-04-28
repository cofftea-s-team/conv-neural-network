#define DEBUG

#include "cnn/config.hpp"
#include "cnn/neural_network.hpp"
#include "cnn/linear.hpp"
#include "cnn/dropout.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <ranges>
#include <chrono>

using std::cout;
using std::endl;

namespace stdr = std::ranges;
namespace stdrv = std::ranges::views;

using namespace std::chrono;

#pragma region better main impl
inline void preload() {
	cout << "Loading...";

	srand(time(NULL));
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

using namespace cnn;

void normalize_input(host::matrix<float>& _Input, float _MinX, float _MaxX, float _MinY, float _MaxY) {
	for (size_t i = 0; i < _Input.rows(); ++i) {
		_Input(i, 0) = (_Input(i, 0) - _MinX) / (_MaxX - _MinX) * 2 - 1.f;
		_Input(i, 1) = (_Input(i, 1) - _MinY) / (_MaxY - _MinY) * 2 - 1.f;
	}
}

void denormalize(host::matrix<float>& _Input, float _MinX, float _MaxX, float _MinY, float _MaxY) {
	for (size_t i = 0; i < _Input.rows(); ++i) {
		_Input(i, 0) = (_Input(i, 0) / 2.f + 1.f) * (_MaxX - _MinX) + _MinX;
		_Input(i, 1) = (_Input(i, 1) / 2.f + 1.f) * (_MaxY - _MinY) + _MinY;
	}
}

auto read_input(const std::string& path) {
	using vt = typename config::value_type;
	std::ifstream file(path);

	host::matrix<vt> input(16, 2);

	while (!file.eof()) {
		for (size_t i = 0; i < input.rows(); ++i) {
			file >> input(i, 0) >> input(i, 1);
		}
		break;
	}
	normalize_input(input, -1.5, 2.5, -1, 2);

	return input;
}

auto read_labels(const std::string& path) {
	using vt = typename config::value_type;
	std::ifstream file(path);

	host::matrix<vt> target(16, 1);

	while (!file.eof()) {
		for (size_t i = 0; i < target.rows(); ++i) {
			file >> target(i, 0);
		}
		break;
	}

	return target;
}

auto read_test(const std::string& path) {
	using vt = typename config::value_type;
	std::ifstream file("grid.txt");

	host::matrix<vt> grid(100000, 2);

	while (!file.eof()) {
		for (size_t i = 0; i < grid.rows(); ++i) {
			file >> grid(i, 0) >> grid(i, 1);
		}
	}
	normalize_input(grid, -1.5, 2.5, -1, 2);

	return grid;
}

void test() {
	auto inputs = read_input("input.txt");
	cuda::matrix<float> inputs2(inputs);

	host::matrix<float> W1(2, 16);
	utils::generate_normal(W1);
	cuda::matrix<float> W2(W1);

	host::vector<float, true> B1(base::shape(1, 16));
	utils::generate_normal(B1);
	cuda::vector<float, true> B2(B1);

	host::matrix<float> W3(16, 1);
	utils::generate_normal(W3);
	cuda::matrix<float> W4(W3);

	host::vector<float, true> B3(base::shape(1, 1));
	utils::generate_normal(B3);
	cuda::vector<float, true> B4(B3);

	host::matrix<float> target = read_labels("labels.txt");
	cuda::matrix<float> target2(target);


	auto r1 = inputs.mul(W1) + B1;
	auto r2 = inputs2.mul(W2) + B2;

	r1.activate<relu>();
	r2.activate<relu>();

	auto r3 = r1.mul(W3) + B3;
	auto r4 = r2.mul(W4) + B4;

	auto error1 = r3 - target;
	auto error2 = r4 - target2;


	auto f1 = r1.T().mul(error1);
	auto f2 = r2.T().mul(error2);

	W3 -= f1;
	W4 -= f2;

	B3 -= error1.sum1();
	B4 -= error2.sum1();

	auto f3 = inputs.T().mul(error1);
	auto f4 = inputs2.T().mul(error2);
//_Error.mul(_Prev_weights->T()) * _Input
	r1.backward<relu>();
	r2.backward<relu>();
	
	auto b1 = error1.mul(W3.T()) * r1; //error
	auto b2 = error2.mul(W4.T()) * r2;

	auto d1 = inputs.T().mul(b1);
	auto d2 = inputs2.T().mul(b2);
	
	W1 -= d1;
	W2 -= d2;

	B1 -= b1.sum1();
	B2 -= b2.sum1();

	std::cout << B1 << B2 << std::endl;

}

// driver code
int _main(std::span<std::string_view> args) {
	test();
	return 1;
	using value_type = typename config::value_type;
	using matrix = typename config::matrix;
	using vector = typename config::vector;
	
	auto inputs = read_input("input.txt");
	auto labels = read_labels("labels.txt");


	neural_network model(
		linear(2, 30),
		relu(),
		linear(30, 20),
		relu(),
		linear(20, 1)
	);
	
	model.learning_rate = 1e-5f;

	auto acc = [&](matrix& output, matrix& target) -> value_type {
		host::matrix<value_type> out = output;
		host::matrix<value_type> tar = target;
		size_t _Total = 0;
		for (size_t i = 0; i < out.size(); ++i) {
			if (round(out.data()[i]) == tar.data()[i]) {
				++_Total;
			}
		}
		return static_cast<value_type>(_Total) / out.size() * 100.;
	};

	matrix in = inputs;
	matrix out = labels;

	model.train(2000, in, out, [&](size_t i, float loss, matrix& m)->void {
		cout << "[" << i << "] acc: " << std::setprecision(3) << acc(m, out) << "%  loss: " << std::setprecision(8) << loss << endl;
	});

	auto grid = read_test("grid.txt");

	matrix test = grid;
	auto res = model.predict(test).to_host();

	std::ofstream file4("output.txt");
	for (size_t i = 0; i < res.rows(); ++i) {
		file4 << res(i, 0) << '\n';
	}
	
	return 0;
}