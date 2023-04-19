#pragma once

#include "../types.hpp"
#include "algebra/avx2_algebra.hpp"
#include "algebra/matrix_mul.hpp"
#include <random>

namespace base {
	template <class _Ty, bool>
	struct allocator;
}

namespace host {
    
	template <class _Ty>
	struct allocator
		: public base::allocator<_Ty, false>
	{
		constexpr _Ty* alloc(size_t _Count) const override {
			return cuda::alloc_paged<_Ty>(_Count);
		}

		constexpr void free(_Ty* ptr) const override {
			cuda::free_paged<_Ty>(ptr);
		}
	};

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



	template <class _Mat, class _Mat2, class _Mat3>
	inline constexpr void matrix_multiply(const _Mat& A, const _Mat2& B, _Mat3& C) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = B.is_transposed();
		size_t N = A.cols();
		size_t M = A.rows();
		size_t N2 = B.cols();
		
		if (N * N2 <= 2048 || M < 32)
			algebra::matrix_multiply<_T1, _T2>(A.data(), B.data(), C.data(), N, M, N2);
		else
			algebra::matrix_multiply_par<_T1, _T2>(A.data(), B.data(), C.data(), N, M, N2);
	}
	
	template <bool _T1, bool _T2, class _Ty>
	inline constexpr void _matrix_add_vector(const _Ty* A, const _Ty* V, _Ty* B, size_t N, size_t M) {
		if (std::is_same_v<_Ty, bfloat16>) {
			
			
		}
		else {
			avx2_add_vector<_T1, _T2>(A, V, B, N, M);
		}
	}

	template <class _Activ_fn, class _Mat>
	inline void activation_apply(_Mat& _M) {
		auto _Data = _M.data();
		for (size_t i = 0; i < _M.size(); ++i) {
			_Data[i] = _Activ_fn::forward(_Data[i]);
		}
	}

	template <class _Mat1, class _Vec, class _Mat2>
	inline constexpr void matrix_add_vector(const _Mat1& A, const _Vec& B, _Mat2& C) {
		constexpr bool _T1 = A.is_transposed();
		constexpr bool _T2 = B.is_transposed();
		size_t N = A.cols();
		size_t M = A.rows();
		_matrix_add_vector<_T1, _T2>(A.data(), B.data(), C.data(), N, M);
	}

	template <class _Mat1, class _Mat2, class _Ty>
	inline constexpr void matrix_add_scalar(const _Mat1& A, _Mat2& B, _Ty _Val) {
		if constexpr (std::is_same_v<_Ty, bfloat16>) {
			for (size_t i = 0; i < A.size(); ++i) {
				B[i] = A[i] + _Val;
			}
		}
		else {
			avx2_add_scalar(A.data(), B.data(), _Val, A.size());
		}
	}

	template <class _Mat1, class _Mat2, class _Ty>
	inline constexpr void matrix_mul_scalar(const _Mat1& A, _Mat2& B, _Ty _Val) {
		if constexpr (std::is_same_v<_Ty, bfloat16>) {
			for (size_t i = 0; i < A.size(); ++i) {
				B[i] = A[i] * _Val;
			}
		}
		else {
			avx2_mul_scalar(A.data(), B.data(), _Val, A.size());
		}
	}

	template <double _Min = -1.0, double _Max = 1.0, class _Mat>
	inline constexpr void fill_normal_distribution(_Mat& _M) {
		using _Ty = typename _Mat::value_type;
		
		constexpr double _Mean = (_Min + _Max) / 2.;
		constexpr double _SD = (_Max - _Mean) / 3.16667; // k=3
		
		std::mt19937 gen(std::random_device{}());
		std::normal_distribution<double> dist(_Mean, _SD);
		
		for (auto&& _El : _M) {
			_Ty _Val = std::max((_Ty)_Min, std::min((_Ty)_Max, (_Ty)dist(gen)));
			_El = _Val;
		}
	}

	template <double _Min = -1.0, double _Max = 1.0, class _Mat>
	inline constexpr void fill_uniform_distribution(_Mat& _M) {
		using _Ty = typename _Mat::value_type;

		std::mt19937 gen(std::random_device{}());
		std::uniform_real_distribution<double> dist(_Min, _Max);

		for (auto&& _El : _M) {
			_Ty _Val = std::max((_Ty)_Min, std::min((_Ty)_Max, (_Ty)dist(gen)));
			_El = _Val;
		}
	}
}