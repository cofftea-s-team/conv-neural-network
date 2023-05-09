#pragma once
#include "../cuda/utils.hpp"
#include "../base/matrix.hpp"
#include "../base/matrix_view.hpp"
#include "algebra/matrix_add.hpp"
#include "algebra/matrix_add_vector.hpp"
#include "algebra/matrix_scalar_mul.hpp"
#include "algebra/matrix_sum.hpp"
#include "algebra/matrix_mul_add_bias.hpp"
#include "vector.hpp"
#include "utils.hpp"
#include <iostream>
#include <concepts>
#include <functional>

namespace cuda {
	template <class _Ty, bool>
	class matrix;
}

namespace host {
	using std::cout;
	using std::endl;
	using std::ostream;

	template <class _Ty, bool _Tr = false>
	class matrix 
		: public base::matrix<_Ty, allocator<_Ty>, _Tr>
	{
		using _Mybase = base::matrix<_Ty, allocator<_Ty>, _Tr>;
	public:
		using iterator = _Ty*;
		using const_iterator = const _Ty*;
		
		using _Mybase::_Mybase;
		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Data;
		
		template <base::allocator_t _Other_all, bool _T2>
		inline matrix(const base::matrix<_Ty, _Other_all, _T2>& _Other)
			: _Mybase(_Other)
		{

		}

		template <base::allocator_t _Other_all, bool _T2>
		inline matrix& operator=(const base::matrix<_Ty, _Other_all, _T2>& _Other) {
			_Mybase::operator=(_Other);
			return *this;
		}

		//
		// matrix multiplication

		template <bool _T>
		inline matrix<_Ty, false> mul(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Cols == _Other.rows());
#endif // !DEBUG
			matrix<_Ty, false> _Res(_Rows, _Other.cols());
			host::matrix_multiply(*this, _Other, _Res);
			return _Res;
		}

		//
		// matrix optimized operations
		
		template <bool _T, bool _T2>
		inline matrix<_Ty, false> mul_add_bias(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other, const base::vector<_Ty, allocator<_Ty>, _T2>& _Vec) const {
#ifdef DEBUG
			assert(_Cols == _Other.rows());
#endif // !DEBUG
			matrix<_Ty, false> _Res(_Rows, _Other.cols());
			host::matrix_mul_add_bias(*this, _Other, _Vec, _Res);
			return _Res;
		}

		//
		// matrix operations
		
		template <bool _T>
		inline matrix& operator+=(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) {
			host::matrix_add<std::plus<_Ty>>(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator-=(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) {
			host::matrix_add<std::minus<_Ty>>(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator/=(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) {
			host::matrix_add<std::divides<_Ty>>(*this, _Other, *this);
			return *this;
		}
		
		template <bool _T>
		inline matrix& operator*=(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) {
			host::matrix_add<std::multiplies<_Ty>>(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator+(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_add<std::plus<_Ty>>(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator-(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_add<std::minus<_Ty>>(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator/(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_add<std::divides<_Ty>>(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator*(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_add<std::multiplies<_Ty>>(*this, _Other, _Res);
			return _Res;
		}

		//
		// vector operations
		
		template <bool _T>
		inline matrix& operator+=(const base::vector<_Ty, host::allocator<_Ty>, _T>& _Other) {
			host::matrix_add_vector<std::plus<_Ty>>(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator-=(const base::vector<_Ty, host::allocator<_Ty>, _T>& _Other) {
			host::matrix_add_vector<std::minus<_Ty>>(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator+(const base::vector<_Ty, host::allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Mybase::shape());
			host::matrix_add_vector<std::plus<_Ty>>(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator-(const base::vector<_Ty, host::allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Mybase::shape());
			host::matrix_add_vector<std::minus<_Ty>>(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix<_Ty, false> operator*(const base::vector<_Ty, host::allocator<_Ty>, _T>& _Other) const {
			matrix<_Ty, false> _Res(_Mybase::shape());
			host::matrix_add_vector<std::multiplies<_Ty>>(*this, _Other, _Res);
			return _Res;
		}
		
		// 
		// scalar operations

		inline matrix& operator+=(const _Ty& _Val) {
			host::matrix_add_scalar(*this, *this, _Val);
			return *this;
		}
		
		inline matrix& operator-=(const _Ty& _Val) {
			host::matrix_add_scalar(*this, *this, -_Val);
			return *this;
		}

		inline matrix& operator*=(const _Ty& _Val) {
			host::matrix_mul_scalar(*this, *this, _Val);
			return *this;
		}

		inline matrix<_Ty, false> operator+(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_add_scalar(*this, _Res, _Val);
			return _Res;
		}

		inline matrix<_Ty, false> operator-(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_add_scalar(*this, _Res, -_Val);
			return _Res;
		}

		inline matrix<_Ty, false> operator*(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_mul_scalar(*this, _Res, _Val);
			return _Res;
		}

		inline matrix<_Ty, false> operator/(const _Ty& _Val) const {
			matrix<_Ty, false> _Res(_Rows, _Cols);
			host::matrix_div_scalar(*this, _Res, _Val);
			return _Res;
		}

		template <activation_fn_t _Fn>
		inline void activate() {
			host::activation_apply<_Fn>(*this);
		}

		template <activation_fn_t _Fn>
		inline void backward() {
			host::backward_apply<_Fn>(*this);
		}

		inline auto sum0() const {
			vector<_Ty, false> _Res(_Rows);
			host::matrix_sum0(*this, _Res);
			return _Res;
		}

		inline auto sum1() const {
			vector<_Ty, true> _Res(base::shape(1, _Cols));
			host::matrix_sum1(*this, _Res);
			return _Res;
		}
		
		inline void dropout(_Ty _Dropout_rate) {
			host::apply_dropout(*this, _Dropout_rate);
		}


		inline auto T() {
			return base::transposed(*this);
		}
		
		inline auto to_cuda() {
			return cuda::matrix<_Ty>(*this);
		}

		inline friend ostream& operator<<(ostream& _Os, const matrix& _M) {
			_Os << "[HOST]\n[" << _M.rows() << "x" << _M.cols() << "] (rows x cols) {\n";
			const _Ty* _Ptr = _M.data();
			for (int i = 0; i < _M.rows(); ++i) {
				_Os << "    ";
				for (int j = 0; j < _M.cols(); ++j) {
					_Os << _Ptr[i * _M.cols() + j] << " ";
				}
				_Os << endl;
			}
			_Os << "}" << endl;
			
			return _Os;
		}
	};
}