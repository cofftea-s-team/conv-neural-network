#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>

using bfloat16 = nv_bfloat16;

namespace cuda {
	template <class _Ty, bool _T1, bool _T2>
	void _matrix_multiply(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M1, size_t M2);


	template <class _Mat, class _Mat2>
	inline void matrix_multiply(const _Mat& A, const _Mat2& B, _Mat& C) {
		using _Ty = typename _Mat::value_type;
		
		size_t N = A.cols();
		size_t M1 = A.rows();
		size_t M2 = B.cols();

		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = B.is_transposed();
		
		_matrix_multiply<_Ty, _T1, _T2>(A.data(), B.data(), C.data(), N, M1, M2);
	}
}