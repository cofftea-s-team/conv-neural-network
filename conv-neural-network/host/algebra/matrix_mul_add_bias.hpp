#pragma once
#include "utils.hpp"
#include "matrix_mul.hpp"

namespace host::algebra {
	
	namespace parallel {

		template <class _Ty>
		inline constexpr void matrix_mul_add_bias(const _Ty* A, const _Ty* B, const _Ty* V, _Ty* C, size_t N, size_t M, size_t N2) {
			parallel_for(M, [&](uint32_t i) {
				for (size_t j = 0; j < N2; ++j) {
					C[i * N2 + j] = V[j] + host::algebra::dot_product<false, false>(A, B, C, N, M, N2, i, j);
				}
				});
		}
	}

	template <class _Ty>
	inline constexpr void matrix_mul_add_bias(const _Ty* A, const _Ty* B, const _Ty* V, _Ty* C, size_t N, size_t M, size_t N2) {
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N2; ++j) {
				C[i * N2 + j] = V[j] + host::algebra::dot_product<false, false>(A, B, C, N, M, N2, i, j);
			}
		}
	}
}

namespace host {

	template <class _Mat, class _Mat2, class _Vec, class _Mat3>
	inline constexpr void matrix_mul_add_bias(const _Mat& A, const _Mat2& B, const _Vec& V, _Mat3& C) {
		size_t N = A.cols();
		size_t M = A.rows();
		size_t N2 = B.cols();

		if (N * N2 <= 2048 || M < 32) {
			algebra::matrix_mul_add_bias(A.data(), B.data(), V.data(), C.data(), N, M, N2);
		}
		else {
			algebra::parallel::matrix_mul_add_bias(A.data(), B.data(), V.data(), C.data(), N, M, N2);
		}
	}
}