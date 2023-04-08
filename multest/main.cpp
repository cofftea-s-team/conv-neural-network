#include <iostream>
#include <iomanip>
#include <Algorithms.h>
#include <chrono>

#define DEBUG
#include "includes.hpp"

using std::cout;
using std::endl;
using namespace std::chrono;

using namespace pipeline;

template <class _Ty>
void print_array(const _Ty* array, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << array[i] << " ";
	}
	std::cout << std::endl;
}

template <class _Ty, size_t _Size>
void fill_array(_Ty(&_Array)[_Size]) {
	for (size_t i = 0; i < _Size; ++i) {
		_Array[i] = (float)2;
	}
}

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cout << std::setprecision(1) << std::fixed;

	host::matrix<bfloat16> A(2, 3);
	for (int i = 0; i < A.rows(); ++i) {
		for (int j = 0; j < A.cols(); ++j) {
			A.data()[i * A.cols() + j] = (float)i + j;
		}
	}

	cuda::matrix<bfloat16> B = A;
	//cuda::matrix<bfloat16> B2(transposed(B)); TODO: Fix this
	//cout << B << endl;
	
	cuda::matrix<bfloat16> C = A;
	
	C = transposed(B);

	cout << B << endl;
	cout << C << endl;
	



}