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
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
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
	py::class_<host::matrix<float>>(m, "matrix_fp32")
		.def(py::init<size_t, size_t>())
		.def("__setitem__", [](host::matrix<float>& _Mat, py::tuple _Idx, float _Val) {
				size_t _I = py::cast<size_t>(_Idx[0]);
				size_t _J = py::cast<size_t>(_Idx[1]);
				_Mat(_I, _J) = _Val;
			})
		.def("__getitem__", [](const host::matrix<float>& _Mat, py::tuple _Idx) { 
				size_t _I = py::cast<size_t>(_Idx[0]);
				size_t _J = py::cast<size_t>(_Idx[1]);
				return _Mat(_I, _J);
			})
		.def("__iter__", [](const host::matrix<float>& _Mat) { return py::make_iterator(_Mat.begin(), _Mat.end()); },
			py::keep_alive<0, 1>())
		.def("__repr__", [](const host::matrix<float>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })
		.def("__str__", [](const host::matrix<float>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })
		;
	py::class_<host::matrix<double>>(m, "matrix_fp64")
		.def(py::init<size_t, size_t>())
		.def("__setitem__", [](host::matrix<double>& _Mat, py::tuple _Idx, double _Val) {
				size_t _I = py::cast<size_t>(_Idx[0]);
				size_t _J = py::cast<size_t>(_Idx[1]);
				_Mat(_I, _J) = _Val;
			})
		.def("__getitem__", [](const host::matrix<double>& _Mat, py::tuple _Idx) { 
				size_t _I = py::cast<size_t>(_Idx[0]);
				size_t _J = py::cast<size_t>(_Idx[1]);
				return _Mat(_I, _J);
			})
		.def("__iter__", [](const host::matrix<double>& _Mat) { return py::make_iterator(_Mat.begin(), _Mat.end()); },
			py::keep_alive<0, 1>())
		.def("__repr__", [](const host::matrix<double>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })
		.def("__str__", [](const host::matrix<double>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })
		;
	py::class_<host::matrix<nv_bfloat16>>(m, "matrix_bf16")
		.def(py::init<size_t, size_t>())
		.def("__setitem__", [](host::matrix<nv_bfloat16>& _Mat, py::tuple _Idx, nv_bfloat16 _Val) {
				size_t _I = py::cast<size_t>(_Idx[0]);
				size_t _J = py::cast<size_t>(_Idx[1]);
				_Mat(_I, _J) = _Val;
			})
		.def("__getitem__", [](const host::matrix<nv_bfloat16>& _Mat, py::tuple _Idx) { 
				size_t _I = py::cast<size_t>(_Idx[0]);
				size_t _J = py::cast<size_t>(_Idx[1]);
				return _Mat(_I, _J);
			})
		.def("__iter__", [](const host::matrix<nv_bfloat16>& _Mat) { return py::make_iterator(_Mat.begin(), _Mat.end()); },
			py::keep_alive<0, 1>())
		.def("__repr__", [](const host::matrix<nv_bfloat16>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })
		.def("__str__", [](const host::matrix<nv_bfloat16>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })
		;
	m.def("train", train<float>);
	m.def("train", train<double>);
	m.def("train", train<nv_bfloat16>);

	m.def("predict", predict<float>);
	m.def("predict", predict<double>);
	m.def("predict", predict<nv_bfloat16>);
}