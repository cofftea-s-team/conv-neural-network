#pragma once
#include <iostream>
#include <iomanip>

#include <chrono>
#include <random>
#include "../cnn/config.hpp"
#include "../cnn/neural_network.hpp"
#include "../cnn/linear.hpp"
#include <string>
#define DEBUG


using std::cout;
using std::endl;
using namespace std::chrono;


//#include "utils.hpp"

//using namespace base;
#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "Python.h"

namespace py = pybind11;

namespace PYBIND11_NAMESPACE {
	namespace detail {
		template <> struct type_caster<float> {
		public:
			PYBIND11_TYPE_CASTER(float, const_name("float"));
			bool load(handle _Source, bool) {
				PyObject* _Src = _Source.ptr();
				PyObject* _Tmp = PyNumber_Float(_Src);
				if (!_Tmp) return false;
				value = PyFloat_AsDouble(_Tmp);
				Py_DECREF(_Tmp);
				if (PyErr_Occurred()) return false;
				return true;
			}
			static handle cast(float _Source, return_value_policy, handle) {
				return PyFloat_FromDouble(_Source);
			}
		};
		template <> struct type_caster<double> {
		public:
			PYBIND11_TYPE_CASTER(double, const_name("float"));
			bool load(handle _Source, bool) {
				PyObject* _Src = _Source.ptr();
				PyObject* _Tmp = PyNumber_Float(_Src);
				if (!_Tmp) return false;
				value = PyFloat_AsDouble(_Tmp);
				Py_DECREF(_Tmp);
				if (PyErr_Occurred()) return false;
				return true;
			}
			static handle cast(double _Source, return_value_policy, handle) {
				return PyFloat_FromDouble(_Source);
			}
		};
	}
}

#include "cuda/memory.hpp"
#include "defines.hpp"

using namespace cnn;

PYBIND11_MODULE(CNN, m) {
	cuda::details::_Data = cuda::alloc<void*>(cuda::details::max_size);

	static neural_network model = {
		linear(2, 30),
		relu(),
		linear(30, 20),
		leaky_relu(),
		linear(20, 1),
	};

	py_nn([&](size_t i, float loss)->void {
		cout << "Epoch: " << i << ", loss: " << loss << endl;
	});

	py_matrix("f", float);
	py_matrix("d", double);
	//py_matrix("bf", bfloat16);

	py::class_<py_nn>(m, "py_nn")
		.def(py::init<>())
		.def("train", &py_nn::train)
		.def("predict", &py_nn::predict);
}

