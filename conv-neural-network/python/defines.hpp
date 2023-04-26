#pragma once
#define DEBUG

#define py_nn(lambda) class py_nn { \
using value_type = typename config::value_type; \
using matrix = typename config::matrix; \
using vector = typename config::vector; \
public: \
	inline void train(size_t _Epochs, matrix& _Input, matrix& _Target, float _Lr) { \
		model.learning_rate = _Lr; \
		model.train(_Epochs, _Input, _Target, lambda); \
	} \
	inline auto predict(matrix& _Input) { \
		return model.predict(_Input); \
	} \
	}

#define py_matrix(name, type) py::class_<host::matrix<type>>(m, (std::string("matrix_") + std::string(name)).c_str(), py::buffer_protocol()) \
.def(py::init<size_t, size_t>()) \
.def(py::init([](py::buffer b) { \
	py::buffer_info info = b.request(); \
	if (info.format != py::format_descriptor<type>::format()) \
		throw std::runtime_error(std::string("Incompatible format: expected a ") + std::string(#type) + std::string(" array!")); \
	if (info.ndim != 2) \
		throw std::runtime_error("Incompatible buffer dimension!"); \
	return new host::matrix<type>( \
		(type*)info.ptr, \
		info.shape[1], \
		info.shape[0] \
	); \
	})) \
	.def_buffer([](host::matrix<type>& _Mat)->py::buffer_info { \
		return py::buffer_info( \
			_Mat.data(), \
			sizeof(type), \
			py::format_descriptor<type>::format(), \
			2, \
			{ _Mat.rows(), _Mat.cols() }, \
			{ sizeof(type) * _Mat.cols(), sizeof(type) } \
		); \
		}) \
.def("__str__", [](const host::matrix<type>& _Mat) { std::stringstream s; s << _Mat; return s.str(); })