#pragma once
#include <iostream>
#include "utils.hpp"
#include "vector.hpp"
#include "algebra/matrix_mul.cuh"
#include "algebra/matrix_add.cuh"
#include "algebra/matrix_sub.cuh"
#include "algebra/matrix_scalar_mul.cuh"
#include "algebra/matrix_add_vector.cuh"
#include "algebra/matrix_sub_vector.cuh"
#include "algebra/matrix_add_scalar.cuh"
#include "algebra/matrix_mul_scalar.cuh"
#include "algebra/matrix_sum.cuh"
#include "algebra/range_reduce.cuh"
#include "activations/activation.cuh"
#include "activations/dropout.cuh"
#include "../cnn/activations.hpp"
#include "../base/matrix.hpp"
#include "../base/types.hpp"
#include "../base/matrix_view.hpp"

namespace host {
	template <class _Ty, bool>
	class matrix;
}

namespace cuda {
	
	template <class _Ty, bool>
	class vector;

	using std::cout;
	using std::endl;
	using std::ostream;

	template <class _Ty>
	struct cuda_to_host_matrix_iterator;

	template <class _Ty, bool _Tr = false>
	class matrix 
		: public base::matrix<_Ty, cuda::allocator<_Ty>, _Tr> 
	{
	protected:
		using _Mybase = base::matrix<_Ty, cuda::allocator<_Ty>, _Tr>;
		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Data;
		using _Mybase::_Al;
	public:
		using _Mybase::_Mybase;
		using iterator = cuda_to_host_matrix_iterator<_Ty>;
		using const_iterator = cuda_to_host_matrix_iterator<const _Ty>;
		
		template <base::allocator_t _Other_all, bool _T2>
		inline matrix& operator=(const base::matrix<_Ty, _Other_all, _T2>& _Other) {
			_Mybase::operator=(_Other);
			return *this;
		}

		//
		// matrix multiplication
		
		template <bool _T>
		inline matrix<_Ty, false> mul(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.rows());
#endif // DEBUG
			matrix<_Ty, false> _Result(_Mybase::rows(), _Other.cols());
			cuda::matrix_multiply(*this, _Other, _Result);
			return _Result;
		}
		
		//
		// matrix operations

		template <bool _T>
		inline matrix& operator+=(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows());
#endif // DEBUG
			cuda::matrix_add(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator-=(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows());
#endif // DEBUG
			cuda::matrix_sub(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator*=(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>&_Other) {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows());
#endif // DEBUG
			cuda::matrix_scalar_mul(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator+(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows());
#endif // DEBUG
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_add(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator-(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows());
#endif // DEBUG
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_sub(*this, _Other, _Res);
			return _Res;
		}
		
		template <bool _T>
		inline matrix<_Ty, false> operator*(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows());
#endif // DEBUG
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_scalar_mul(*this, _Other, _Res);
			return _Res;
		}

		//
		// vector operations
		
		template <bool _T>
		inline matrix& operator+=(const base::vector<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
			cuda::matrix_add_vector(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator-=(const base::vector<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
			cuda::matrix_sub_vector(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator+(const base::vector<_Ty, cuda::allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.size() || _Mybase::rows() == _Other.size());
#endif // DEBUG
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_add_vector(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator-(const base::vector<_Ty, cuda::allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.size() || _Mybase::rows() == _Other.size());
#endif // DEBUG
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_sub_vector(*this, _Other, _Res);
			return _Res;
		}

		//
		// scalar operations

		inline matrix& operator+=(const _Ty& _Val) {
			cuda::matrix_add_scalar(*this, *this, _Val);
			return *this;
		}

		inline matrix& operator-=(const _Ty& _Val) {
			cuda::matrix_add_scalar(*this, *this, -_Val);
			return *this;
		}

		inline matrix& operator*=(const _Ty& _Val) {
			cuda::matrix_mul_scalar(*this, *this, _Val);
			return *this;
		}

		inline matrix<_Ty, false> operator+(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_add_scalar(*this, _Res, _Val);
			return _Res;
		}

		inline matrix<_Ty, false> operator-(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_add_scalar(*this, _Res, _Val);
			return _Res;
		}

		inline matrix<_Ty, false> operator*(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Mybase::shape());
			cuda::matrix_mul_scalar(*this, _Res, _Val);
			return _Res;
		}

		template <activation_fn_t _Fn>
		inline void activate() {
			cuda::forward_apply<_Fn>(*this);
		}

		template <activation_fn_t _Fn>
		inline void backward() {
			cuda::backward_apply<_Fn>(*this);
		}

		inline vector<_Ty> sum0() const {
			vector<_Ty> _Res(_Mybase::rows());
			cuda::matrix_sum(*this, _Res);
			return _Res;
		}

		inline void dropout(_Ty _Dropout_rate) {
			cuda::apply_dropout(*this, _Dropout_rate);
		}
		
		inline auto T() {
			return base::transposed(*this);
		}

		inline friend ostream& operator<<(ostream& _Os, const matrix& _M) {
			_Os << "[CUDA]\n[" << _M.rows() << "x" << _M.cols() << "] (rows x cols) {\n";
			const _Ty* _Ptr = _M.data();
			for (int i = 0; i < _M.rows(); ++i) {
				_Os << "    ";
				for (int j = 0; j < _M.cols(); ++j) {
					_Os << cuda::from_cuda(&_Ptr[i * _M.cols() + j]) << " ";
				}
				_Os << endl;
			}
			_Os << "}" << endl;

			return _Os;
		}

		inline iterator begin() {
			return iterator(_Data);
		}

		inline iterator end() {
			return iterator(_Data + _Rows * _Cols);
		}
		
		inline const_iterator begin() const {
			return const_iterator(_Data);
		}

		inline const_iterator end() const {
			return const_iterator(_Data + _Rows * _Cols);
		}
		
		inline auto to_host() {
			return host::matrix<_Ty>(*this);
		}

	};
}