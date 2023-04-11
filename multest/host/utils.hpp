#pragma once

#include "../types.hpp"
#include "avx2_algebra.hpp"
#include <random>

namespace host {
    
	template <class _Ty>
    inline constexpr void copy_and_transpose(const _Ty* _Src, _Ty* _Dst, size_t _New_rows, size_t _New_cols) {
		for (size_t i = 0; i < _New_rows; ++i) {
			for (size_t j = 0; j < _New_cols; ++j) {
				_Dst[i * _New_cols + j] = _Src[j * _New_rows + i];
			}
		}
    }

	template <>
	inline void copy_and_transpose<float>(const float* _Src, float* _Dst, size_t _New_rows, size_t _New_cols) {
		avx2_transpose(_Src, _Dst, _New_rows, _New_cols);
	}

	template <class _Mat, class _Mat2>
	inline constexpr void matrix_multiply(const _Mat& A, const _Mat2& B, _Mat& C) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = B.is_transposed();
		size_t N = A.cols();
		size_t M1 = A.rows();
		size_t M2 = B.cols();
		_matrix_multiply<_T1, _T2>(A.data(), B.data(), C.data(), N, M1, M2);
	}
	
	template <bool _T1, bool _T2, class _Ty>
	inline constexpr void _matrix_multiply(const _Ty* A, const _Ty* B, _Ty* C, size_t N, size_t M1, size_t M2) {
		for (size_t i = 0; i < M1; i++) {
			for (size_t j = 0; j < M2; j++) {
				_Ty _Sum = 0.;
				for (size_t k = 0; k < N; k++) {
					if constexpr (_T2) {
						if constexpr (_T1) 
							_Sum = _Sum + A[k * M1 + i] * B[j * N + k];
						else 
							_Sum = _Sum + A[i * N + k] * B[j * N + k];
					}
					else {
						if constexpr (_T1) 
							_Sum = _Sum + A[k * M1 + i] * B[k * M2 + j];
						else 
							_Sum = _Sum + A[i * N + k] * B[k * M2 + j];						
					}
				}
				C[i * M2 + j] = _Sum;
			}
		}
	}

	template <class _Mat, double _Min = -1.0, double _Max = 1.0>
	inline constexpr void fill_normal_distribution(_Mat& _Matrix) {
		using _Ty = typename _Mat::value_type;
		
		constexpr double _Mean = (_Min + _Max) / 2.;
		constexpr double _SD = (_Max - _Mean) / 3.16667; // k=3
		
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(_Mean, _SD);
		
		for (size_t i = 0; i < _Matrix.size(); ++i) {
			_Ty _Val = std::max((_Ty)_Min, std::min((_Ty)_Max, (_Ty)dist(gen)));
			_Matrix.data()[i] = _Val;
		}
	}
}