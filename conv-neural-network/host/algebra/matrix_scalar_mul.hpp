#pragma once
#include "utils.hpp"
#include <immintrin.h>

namespace host::algebra {

	namespace parallel {
		template <bool _T1, bool _T2, class _Ty>
		inline constexpr void matrix_scalar_mul(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M) {
			parallel_for(M, [&](size_t i) {
				for (size_t j = 0; j < N; ++j) {
					if constexpr (_T1 && _T2) {
						C[i * N + j] = A[j * M + i] * B[j * M + i];
					}
					else if constexpr (_T1 && !_T2) {
						C[i * N + j] = A[j * M + i] * B[i * N + j];
					}
					else if	constexpr (!_T1 && _T2) {
						C[i * N + j] = A[i * N + j] * B[j * M + i];
					}
					else {
						C[i * N + j] = A[i * N + j] * B[i * N + j];
					}
				}
			});
		}
	}

	template <bool _T1, bool _T2, class _Ty>
	inline constexpr void matrix_scalar_mul(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M) {
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				if constexpr (_T1 && _T2) {
					C[i * N + j] = A[j * M + i] * B[j * M + i];
				}
				else if constexpr (_T1 && !_T2) {
					C[i * N + j] = A[j * M + i] * B[i * N + j];
				}
				else if	constexpr (!_T1 && _T2) {
					C[i * N + j] = A[i * N + j] * B[j * M + i];
				}
				else {
					C[i * N + j] = A[i * N + j] * B[i * N + j];
				}
			}
		}
	}
}

namespace host {

	template <class _Mat1, class _Mat2, class _Mat3>
	inline constexpr void matrix_scalar_mul(const _Mat1& A, const _Mat2& B, _Mat3& C) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = B.is_transposed();
		size_t N = A.cols();
		size_t M = A.rows();

		if (N * M <= 4096) {
			host::algebra::matrix_scalar_mul<_T1, _T2>(A.data(), B.data(), C.data(), N, M);
		}
		else {
			host::algebra::parallel::matrix_scalar_mul<_T1, _T2>(A.data(), B.data(), C.data(), N, M);
		}
	}
}