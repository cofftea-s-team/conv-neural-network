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

	struct sigmoid {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return 1. / (1. + exp(-x));
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

	struct bsigmoid {
		template <class _Ty>
		__device__ __host__ inline static _Ty forward(_Ty x) {
			return x * (1. - x);
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x * (1. - x);
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
			return ::exp(x) / sum;
		}

		template <class _Ty>
		__device__ __host__ inline static _Ty backward(_Ty x) {
			return x * (1. - x);
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
}