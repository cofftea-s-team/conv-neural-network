#define DEBUG

#include "cnn/config.hpp"
#include "cnn/optimizers.hpp"
#include "cnn/neural_network.hpp"
#include "cnn/linear.hpp"
#include "cnn/dropout.hpp"
#include "cnn/loss.hpp"
#include "cnn/file.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <string_view>
#include <filesystem>
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
inline void preload(std::span<std::string_view> args) {
	std::cout << "Loading...";

	srand(time(NULL));
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(NULL);
	std::cout << std::setprecision(7) << std::fixed;
	
	// force cuda to load
	cuda::matrix<float> f(16, 16);
	f.mul(transposed(f));

	cnn::file::working_directory = std::filesystem::path(args[0]).parent_path();
	
	std::cout << " done!" << std::endl;
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

	preload(args);
	int exit_code = _main(args);

	return exit_code;
}
#pragma endregion

#define TIMER_START(x) auto _s##x = high_resolution_clock::now(); auto _1s##x = clock();
#define TIMER_END(x) auto _s2##x = high_resolution_clock::now(); auto _1s2##x = clock();
#define TIMER_RESULT(x, pre) auto count1##x = duration_cast<milliseconds>(_s2##x - _s##x).count(); auto count2##x = (_1s2##x - _1s##x); cout << '[' << ##pre << "]\n" << "user time: " << count1##x << "ms\nsys time: " << count2##x << "ms\n" << endl;

using value_type = typename cnn::config::value_type;
using matrix = typename cnn::config::matrix;
using dual_matrix = typename cnn::config::dual_matrix;
using vector = typename cnn::config::vector;


std::vector<host::matrix<value_type>> read_input(std::string_view path, size_t lines) {
	std::vector<host::matrix<value_type>> result;

	std::ifstream file(path.data());
	if (!file) {
		std::cerr << "File not found: " << path << std::endl;
		return result;
	}

	while (lines--) {
		host::matrix<value_type> m(28, 28);
		for (size_t i = 0; i < 28; ++i) {
			for (size_t j = 0; j < 28; ++j) {
				file >> m(i, j);
				m(i, j) /= 255.0;
			}
		}
		result.push_back(std::move(m));
	}

	return result;
}

std::vector<size_t> read_labels(std::string_view path, size_t lines) {
	std::vector<size_t> result;

	std::ifstream file(path.data());

	while (lines--) {
		size_t label;
		file >> label;
		result.push_back(label);
	}

	return result;
}

matrix create_batch(const std::vector<host::matrix<value_type>>& data, size_t start_idx, size_t batch_size) {
	matrix result(batch_size, 784);

	for (size_t i = 0; i < batch_size; ++i) {
		if constexpr (matrix::alloc::is_cuda())
			cuda::memcpy(data[start_idx + i].data(), &result.data()[i * 784], 784, HostToDevice);
		else
			cuda::memcpy(data[start_idx + i].data(), &result.data()[i * 784], 784, HostToHost);
	}

	return result;
}

matrix create_batch_labels(const std::vector<size_t>& data, size_t start_idx, size_t batch_size) {
	host::matrix<value_type> result(batch_size, 10);

	for (size_t i = 0; i < batch_size; ++i) {
		result.data()[i * 10 + data[start_idx + i]] = 1.;
	}

	return matrix(result);
}

using namespace cnn;

void print_batch(const matrix& matrix, size_t id) {
	std::cout << std::setprecision(1);
	std::cout << "Batch " << id << ":\n";
	for (size_t i = 0; i < matrix.cols(); ++i) {
		std::cout << cuda::from_cuda( &matrix.data()[id * matrix.cols() + i] )<< ' ';
		if (i % 28 == 27) {
			std::cout << '\n';
		}
	}
	std::cout << '\n';
}

// driver code
int _main(std::span<std::string_view> args) {
	const std::string_view train_file = "../train-images.txt";
	const std::string_view train_label_file = "../train-labels.txt";
	const std::string_view test_file = "../test-images.txt";
	const std::string_view test_label_file = "../train-labels.txt";

	static constexpr size_t batch_size = 1024;
	static constexpr size_t data_count = 8144;
	static constexpr size_t N = ((data_count - 1) | (batch_size - 1)) + 1;

	static constexpr size_t iterations = 50;
	static constexpr size_t batch_count = N / batch_size;
	
	auto inputs = read_input(train_file, N);
	auto labels = read_labels(train_label_file, N);
	
	neural_network nn(
		linear(784, 512),
		relu(),
		linear(512, 128),
		relu(),
		linear(128, 10),
		softmax()
	);
	
	hyperparameters<adam> params = {
		.learning_rate = 0.001,
		.beta1 = 0.9,
		.beta2 = 0.999,
	};
	
	adam optimizer(nn.linear_count, params);
	
	for (size_t epoch = 0; epoch < 7; ++epoch) {	
		
		for (size_t i = 0; i < batch_count; ++i) {
			matrix batch = create_batch(inputs, i * batch_size, batch_size);
			matrix batch_labels = create_batch_labels(labels, i * batch_size, batch_size);

			logger log(100, [=](size_t it, value_type loss, value_type acc) {
				std::cout << "epoch:" << std::setw(3) << epoch << ",  it:" << std::setw(5) << (it + i * iterations) << '/' << batch_count * iterations\
					<< ",  loss: " << loss << "  acc: " << acc << " " << std::endl;
			});

			nn.train<cross_entropy>(iterations, batch, batch_labels, optimizer, log);
		}
	}
	
	//file f("/model");
	//f.load(nn);

	matrix batch = create_batch(inputs, 0, 1024);
	matrix batch_labels = create_batch_labels(labels, 0, 1024);
	
	auto preds = nn.predict(batch);
	std::cout << accuracy(preds, batch_labels) << std::endl;
	
	auto idx = argmax(preds);
	for (size_t i = 0; i < 32; ++i)
		std::cout << idx[i] << ' ';
	std::cout << std::endl;
	for (size_t i = 0; i < 32; ++i)
		std::cout << labels[i] << ' ';
	std::cout << std::endl;

	return 0;
}