#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudadevrt.lib")


#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}
PYBIND11_MODULE(pytest, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
