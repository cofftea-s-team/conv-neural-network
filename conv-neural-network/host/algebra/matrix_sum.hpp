#pragma once
#include "utils.hpp"

namespace host::algebra {

	namespace parallel {
	
		template <bool _T1, class _Ty>
		inline void matrix_sum0(const _Ty* A, _Ty* V, size_t N, size_t M) {
			parallel_for (M, [&](size_t i) {
				_Ty _Sum = 0.f;
				for (size_t j = 0; j < N; ++j) {
					if constexpr (_T1)
						_Sum += A[j * M + i];
					else
						_Sum += A[i * N + j];
				}
				V[i] = _Sum;
			});
		}

		template <bool _T1, class _Ty>
		inline void matrix_sum1(const _Ty* A, _Ty* V, size_t N, size_t M) {
			parallel_for(N, [&](size_t j) {
				_Ty _Sum = 0.f;
				for (size_t i = 0; i < M; ++i) {
					if constexpr (_T1)
						_Sum += A[j * M + i];
					else
						_Sum += A[i * N + j];
				}
				V[j] = _Sum;
			});
		}
	}

	template <bool _T1, class _Ty>
	inline void matrix_sum0(const _Ty* A, _Ty* V, size_t N, size_t M) {
		for (size_t i = 0; i < M; ++i) {
			_Ty _Sum = 0.f;
			for (size_t j = 0; j < N; ++j) {
				if constexpr (_T1)
					_Sum += A[j * M + i];
				else
					_Sum += A[i * N + j];
			}
			V[i] = _Sum;
		}
	}

	template <bool _T1, class _Ty>
	inline void matrix_sum1(const _Ty* A, _Ty* V, size_t N, size_t M) {
		for (size_t j = 0; j < N; ++j) {
			_Ty _Sum = 0.f;
			for (size_t i = 0; i < M; ++i) {
				if constexpr (_T1)
					_Sum += A[j * M + i];
				else
					_Sum += A[i * N + j];
			}
			V[j] = _Sum;
		}
	}
}

namespace host {

	template <class _Mat, class _Vec>
	inline constexpr void matrix_sum0(const _Mat& A, _Vec& V) {
		static constexpr bool _T = A.is_transposed();
		size_t N = A.cols();
		size_t M = A.rows();

		if (N * M <= 4096) {
			host::algebra::matrix_sum0<_T>(A.data(), V.data(), N, M);
		}
		else {
			host::algebra::parallel::matrix_sum0<_T>(A.data(), V.data(), N, M);
		}
	}

	template <class _Mat, class _Vec>
	inline constexpr void matrix_sum1(const _Mat& A, _Vec& V) {
		static constexpr bool _T = A.is_transposed();
		size_t N = A.cols();
		size_t M = A.rows();

		if (N * M <= 4096) {
			host::algebra::matrix_sum1<_T>(A.data(), V.data(), N, M);
		}
		else {
			host::algebra::parallel::matrix_sum1<_T>(A.data(), V.data(), N, M);
		}
	}
}