#include <iostream>
#include <iomanip>

#include <chrono>
#include <random>
#include "host/utils.hpp"
#define DEBUG


using std::cout;
using std::endl;
using namespace std::chrono;

#include "cuda/activations/activation.cuh"
#include "utils.hpp"

using namespace base;
#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "Python.h"
#include "train.hpp"

namespace py = pybind11;

namespace PYBIND11_NAMESPACE {
	namespace detail {
		template <> struct type_caster<nv_bfloat16> {
		public:
			PYBIND11_TYPE_CASTER(nv_bfloat16, const_name("nv_bfloat16"));
			bool load(handle _Source, bool) {
				PyObject* _Src = _Source.ptr();
				PyObject* _Tmp = PyNumber_Float(_Src);
				if (!_Tmp) return false;
				value = PyFloat_AsDouble(_Tmp);
				Py_DECREF(_Tmp);
				if (PyErr_Occurred()) return false;
				return true;
			}
			static handle cast(nv_bfloat16 _Source, return_value_policy, handle) {
				return PyFloat_FromDouble(_Source);
			}
		};
		template <> struct type_caster<double> {
		public:
			PYBIND11_TYPE_CASTER(double, const_name("nv_bfloat16"));
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

PYBIND11_MODULE(CNN, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring
}