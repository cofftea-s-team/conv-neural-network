#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"

using bfloat16 = nv_bfloat16;

namespace cuda {
	
	template <class _Ty>
	void _matrix_mul_add_bias(const _Ty* A, const _Ty* B, const _Ty* V, _Ty* C, size_t N, size_t M1, size_t M2);

	template <class _Mat, class _Mat2, class _Vec, class _Mat3>
	inline void matrix_mul_add_bias(const _Mat& A, const _Mat2& B, const _Vec& V, _Mat3& C) {
		using _Ty = typename _Mat::value_type;

		size_t N = A.cols();
		size_t M1 = A.rows();
		size_t M2 = B.cols();

		_matrix_mul_add_bias(A.data(), B.data(), V.data(), C.data(), N, M1, M2);
	}
}