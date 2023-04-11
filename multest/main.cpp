#include <iostream>
#include <iomanip>
#include <Algorithms.h>
#include <chrono>
#include <random>
#include "host/utils.hpp"
#define DEBUG
#include "includes.hpp"

using std::cout;
using std::endl;
using namespace std::chrono;

using namespace pipeline;
#include "cuda/activations/activation.cuh"
#include "utils.hpp"

using namespace base;

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cout << std::setprecision(3) << std::fixed;

	host::matrix<bfloat16> _A(3, 3);
	fill(_A.begin(), _A.end(), 1.);


	cuda::matrix<bfloat16> A = _A;

	_A.data()[2] = 2.;
	_A.data()[1] = 2.;
	cuda::matrix<bfloat16> B = _A;
	
	
	//A += transposed(B);
	A += B;

	cout << A << endl;

	cout << A.mul(B) << endl;
	
	

	
}
