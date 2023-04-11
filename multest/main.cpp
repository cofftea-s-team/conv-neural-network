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




void train() {

}

