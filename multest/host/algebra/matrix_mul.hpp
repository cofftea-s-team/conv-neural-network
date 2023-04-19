#pragma once
#include "utils.hpp"

namespace host::algebra {
	
	template <bool _T1, bool _T2, class _Ty>
	inline constexpr _Ty dot_product(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M, size_t N2, size_t i, size_t j) {
		_Ty _Sum = 0.f;
		for (size_t k = 0; k < N; ++k) {
			if constexpr (_T2) {
				if constexpr (_T1)
					_Sum = _Sum + A[k * M + i] * B[j * N + k];
				else
					_Sum = _Sum + A[i * N + k] * B[j * N + k];
			}
			else {
				if constexpr (_T1)
					_Sum = _Sum + A[k * M + i] * B[k * N2 + j];
				else
					_Sum = _Sum + A[i * N + k] * B[k * N2 + j];
			}
		}
		return _Sum;
	}

	namespace parallel {
		
		template <bool _T1, bool _T2, class _Ty>
		inline constexpr void matrix_multiply(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M, size_t N2) {
			parallel_for(M, [&](uint32_t i) {
				for (size_t j = 0; j < N2; ++j) {
					C[i * N2 + j] = host::algebra::dot_product<_T1, _T2>(A, B, C, N, M, N2, i, j);
				}
			});
		}
	}

	template <bool _T1, bool _T2, class _Ty>
	inline constexpr void matrix_multiply(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M, size_t N2) {
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N2; ++j) {
				C[i * N2 + j] = host::algebra::dot_product<_T1, _T2>(A, B, C, N, M, N2, i, j);
			}
		}
	}
}

namespace host {
	
	template <class _Mat, class _Mat2, class _Mat3>
	inline constexpr void matrix_multiply(const _Mat& A, const _Mat2& B, _Mat3& C) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = B.is_transposed();
		size_t N = A.cols();
		size_t M = A.rows();
		size_t N2 = B.cols();

		if (N * N2 <= 2048 || M < 32) {
			algebra::matrix_multiply<_T1, _T2>(A.data(), B.data(), C.data(), N, M, N2);
		}
		else {
			algebra::parallel::matrix_multiply<_T1, _T2>(A.data(), B.data(), C.data(), N, M, N2);
		}
	}
}