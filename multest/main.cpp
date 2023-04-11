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

	auto sh1 = shape(16, 32);
	auto sh2 = shape(32, 64);

	
	cuda::dual_matrix<float> A(sh1);
	cuda::matrix<float> B(sh2);
	
	shape res_sh = get_mul_shape(A, B);

	cout << "A: " << A.shape() << endl;
	cout << "B: " << B.shape() << endl;
	cout << "AxB: " << res_sh << endl;

	A.alloc_result(res_sh);

	auto res = A.mul(B);
	auto res2 = static_cast<cuda::matrix<float>>(A).mul(B);
	
	cout << "AxB: " << res.shape() << endl;
	cout << "AxB: " << res2.shape() << endl;

	
}