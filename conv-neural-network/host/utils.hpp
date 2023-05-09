#pragma once

#include <execution>
#include <algorithm>
#include "../base/types.hpp"
#include "algebra/avx2_algebra.hpp"
#include "algebra/matrix_mul.hpp"
#include "../cuda/utils.hpp"
#include <random>
#include "../cnn/activations.hpp"

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
	
	template <bool _T1, bool _T2, class _Ty>
	inline constexpr void _matrix_add_vector(const _Ty* A, const _Ty* V, _Ty* B, size_t N, size_t M) {
		if (std::is_same_v<_Ty, bfloat16>) {
			
			
		}
		else {
			avx2_add_vector<_T1, _T2, true>(A, V, B, N, M);
		}
	}

	template <class _Activ_fn, class _Mat>
	inline void activation_apply(_Mat& _M) {	
		if constexpr (std::is_same_v<_Activ_fn, cnn::softmax>) {
			activation_apply_softmax(_M);
		}
		else {
			auto _Data = _M.data();
			for (size_t i = 0; i < _M.size(); ++i) {
				_Data[i] = _Activ_fn::forward(_Data[i]);
			}
		}
	}

	template <class _Mat>
	inline void activation_apply_softmax(_Mat& _M) {
		activation_apply<cnn::exp>(_M);
		auto _Data = _M.data();
		for (size_t i = 0; i < _M.rows(); ++i) {
			auto _Sum = std::reduce(std::execution::par_unseq, _Data + i, _Data + i + _M.cols());
			for (size_t j = 0; j < _M.cols(); ++j)	
				_Data[i * _M.cols() + j] = cnn::softmax::forward(_Data[i * _M.cols() + j], _Sum);
		}
	}

	template <class _Activ_fn, class _Mat>
	inline void backward_apply(_Mat& _M) {
		auto _Data = _M.data();
		for (size_t i = 0; i < _M.size(); ++i) {
			_Data[i] = _Activ_fn::backward(_Data[i]);
		}
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

	template <class _Mat1, class _Mat2, class _Ty>
	inline constexpr void matrix_div_scalar(const _Mat1& A, _Mat2& B, _Ty _Val) {
		if constexpr (std::is_same_v<_Ty, bfloat16>) {
			for (size_t i = 0; i < A.size(); ++i) {
				B[i] = A[i] / _Val;
			}
		}
		else {
			avx2_div_scalar(A.data(), B.data(), _Val, A.size());
		}
	}

	template <double _Min = -1.0, double _Max = 1.0, class _Mat>
	inline constexpr void fill_normal_distribution(_Mat& _M) {
		using _Ty = typename _Mat::value_type;
		
		constexpr double _Mean = (_Min + _Max) / 2.;
		constexpr double _SD = (_Max - _Mean) / 3; // k=3
		
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

	template <class _Mat, class _Ty = typename _Mat::value_type>
	inline constexpr void apply_dropout(_Mat& _M, _Ty _Dropout_rate) {
		std::mt19937 gen(std::random_device{}());
		std::uniform_real_distribution<_Ty> dist(0.0, 1.0);

		for (auto&& _El : _M) {
			if (dist(gen) < _Dropout_rate) {
				_El = 0.f;
			}
		}
	}
}