#define DEBUG
#include "cuda/matrix.hpp"
#include "host/matrix.hpp"
#include "cuda/dual_matrix.hpp"

#include <iostream>
#include <iomanip>
#include <Algorithms.h>
#include <chrono>
#include <random>
#include <vector.h>

using std::cout;
using std::endl;
using namespace std::chrono;

using namespace pipeline;
#include "utils.hpp"

using namespace base;

inline auto force_load_cuda() {
	cout << "Loading..." << endl;
	cuda::matrix<float> f(16, 16);
	fill_normal_distribution(f);
	auto x = f.mul(transposed(f)).to_host() | sum;
	cout << "Cuda loaded!" << endl;
	return x;
}

int main() {


	std::ios_base::sync_with_stdio(false);
	std::cout << std::setprecision(3) << std::fixed;
	force_load_cuda();

	auto sh1 = shape(16, 16);
	auto sh2 = shape(16, 16);

	cuda::dual_matrix<float> A(sh1);
	cuda::matrix<float> B(sh2);
	
	A.alloc_result(get_mul_shape(A, B));

	auto C = A.mul(B);

	A = A.mul(C);
	
	auto C1 = A.mul(transposed(A));
	
	A = A.mul(C1);
	
}
