#define DEBUG
#include "cuda/matrix.hpp"
#include "host/matrix.hpp"
#include "host/vector.hpp"
#include "cuda/dual_matrix.hpp"
#include "cuda/vector.hpp"

#include "vector_view.hpp"
#include <iostream>
#include <iomanip>
//#include <Algorithms.h>
#include <chrono>
#include <random>


using std::cout;
using std::endl;
using namespace std::chrono;

#include "utils.hpp"

using namespace base;

void force_load_cuda() {
	cout << "Loading..." << endl;
	cuda::matrix<float> f(16, 16);
	f.mul(f);
	cout << "Cuda loaded!" << endl;
}

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cout << std::setprecision(3) << std::fixed;
	force_load_cuda();

	auto sh1 = shape(16, 24);
	host::matrix<float> A1(sh1);
	for (int i = 0; i < A1.size(); ++i) {
		A1.data()[i] = 0;// i / (float)A1.cols();
	}
	host::vector<float> v1(sh1.rows());
	for (int i = 0; i < v1.size(); ++i) {
		v1.data()[i] = i;
	}
	cout << A1 << endl;
	cout << v1 << endl;
	A1 += v1;
	cout << A1 << endl;

}
