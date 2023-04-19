#pragma once
#include "utils.hpp"
#include <immintrin.h>

namespace host::algebra {
	
	namespace parallel {
		template <class _Ty>
		inline constexpr void matrix_transpose(const _Ty* A, _Ty* B, size_t N, size_t M) {
			parallel_for(M, [&](size_t i) {
				for (size_t j = 0; j < N; ++j) {
					B[i * N + j] = A[j * M + i];
				}
			});
		}
	}
	
	template <class _Ty>
	inline constexpr void matrix_transpose(const _Ty* A, _Ty* B, size_t N, size_t M) {
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				B[i * N + j] = A[j * M + i];
			}
		}
	}
}

namespace host {
	
	template <class _Mat1, class _Mat2>
	inline constexpr void matrix_transpose(const _Mat1& A, _Mat2& B) {
		size_t N = A.cols();
		size_t M = A.rows();
		
		if (N * M <= 4096) {
			algebra::matrix_transpose(A.data(), B.data(), N, M);
		}
		else {
			algebra::parallel::matrix_transpose(A.data(), B.data(), N, M);
		}
	}
}