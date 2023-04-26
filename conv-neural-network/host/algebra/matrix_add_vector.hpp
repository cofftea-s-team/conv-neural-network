#pragma once
#include "utils.hpp"
#include <immintrin.h>

namespace host::algebra {

	namespace parallel {

		template <bool _T1, bool _T2, class _Ty, class _Pred>
		inline constexpr void matrix_add_vector(const _Ty* A, const _Ty* V, _Ty* B, size_t N, size_t M, _Pred _Fn) {
			parallel_for (M, [&](size_t i) {
				for (size_t j = 0; j < N; ++j) {
					if constexpr (_T1 && _T2) {
						B[i * N + j] = _Fn(A[i * N + j], V[j]);
					}
					else if constexpr (_T1 && !_T2) {
						B[i * N + j] = _Fn(A[i * N + j], V[i]);
					}
					else if constexpr (!_T1 && _T2) {
						B[i * N + j] = _Fn(A[i * N + j], V[j]);
					}
					else {
						B[i * N + j] = _Fn(A[i * N + j], V[i]);
					}
				}
			});
		}
	}

	template <bool _T1, bool _T2, class _Ty, class _Pred>
	inline constexpr void matrix_add_vector(const _Ty* A, const _Ty* V, _Ty* B, size_t N, size_t M, _Pred _Fn) {
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				if constexpr (_T1 && _T2) {
					B[i * N + j] = _Fn(A[i * N + j], V[j]);
				}
				else if constexpr (_T1 && !_T2) {
					B[i * N + j] = _Fn(A[i * N + j], V[i]);
				}
				else if constexpr (!_T1 && _T2) {
					B[i * N + j] = _Fn(A[i * N + j], V[j]);
				}
				else {
					B[i * N + j] = _Fn(A[i * N + j], V[i]);
				}
			}
		}
	}
}

namespace host {
	
	template <class _Pred, class _Mat1, class _Vec, class _Mat2>
	inline constexpr void matrix_add_vector(const _Mat1& A, const _Vec& V, _Mat2& B) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = V.is_transposed();
		
		size_t N = A.cols();
		size_t M = A.rows();

		if (N * M <= 4096) {
			algebra::matrix_add_vector<_T1, _T2>(A.data(), V.data(), B.data(), N, M, _Pred{});
		}
		else {
			algebra::parallel::matrix_add_vector<_T1, _T2>(A.data(), V.data(), B.data(), N, M, _Pred{});
		}
	}
}