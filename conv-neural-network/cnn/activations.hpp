#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

namespace cnn {
	struct relu {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return x > 0. ? x : 0.;
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x > 0. ? 1. : 0.;
		}
	};

	struct relu1 {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return x > 0. ? (x < 1. ? x : 1.) : 0.;
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x > 0. ? (x < 1. ? 1. : 0.) : 0.;
		}
	};

	struct sigmoid {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return 1. / (1. + ::exp(-x));
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x * (1. - x);
		}
	};

	struct tanh {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return ::tanh(x);
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return 1. - x * x;
		}
	};

	struct leaky_relu {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return x > 0. ? x : 0.01 * x;
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x > 0. ? 1. : 0.01;
		}
	};

	struct softmax {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x, _Ty sum) {
			return x / sum;
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x * (1. - x);
		}
	};

	struct exp {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return ::exp(x);
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x;
		}
	};

	struct log {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return ::log(x);
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return 1. / x;
		}
	};

	struct sqrt {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return ::sqrt(x);
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return 1. / (2. * ::sqrt(x));
		}
	};

	struct identity {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return x;
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return 1.;
		}
	};
}