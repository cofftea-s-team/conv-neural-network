#pragma once
#include <iostream>
#include "utils.hpp"
#include "vector.hpp"
#include "algebra/matrix_mul.cuh"
#include "algebra/matrix_add.cuh"
#include "algebra/matrix_sub.cuh"
#include "algebra/matrix_add_vector.cuh"
#include "algebra/matrix_sub_vector.cuh"
#include "algebra/matrix_add_scalar.cuh"
#include "activations/activation.cuh"
#include "../activations.hpp"
#include "../matrix.hpp"
#include "../types.hpp"
#include "../matrix_view.hpp"

namespace host {
	template <class _Ty>
	class matrix;
}

namespace cuda {
	
	using std::cout;
	using std::endl;
	using std::ostream;

	template <class _Ty>
	struct cuda_to_host_matrix_iterator;

	template <class _Ty>
	class matrix 
		: public base::matrix<_Ty, cuda::allocator<_Ty>, false> 
	{
	protected:
		using _Mybase = base::matrix<_Ty, cuda::allocator<_Ty>, false>;
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

		template <bool _T>
		inline matrix mul(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
#ifdef DEBUG
			assert(_Mybase::cols() == _Other.rows());
#endif // DEBUG
			matrix _Result(_Mybase::rows(), _Other.cols());
			//cuda::matrix_multiply(_Mybase::data(), _Other.data(), _Result.data(), _Mybase::cols(), _Mybase::rows(), _Other.cols());
			using base_type = base::matrix<_Ty, cuda::allocator<_Ty>, false>&;
			cuda::matrix_multiply(*this, _Other, _Result);
			return _Result;
		}

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

		inline matrix& operator+=(const _Ty& _Val) {
			cuda::matrix_add_scalar(*this, *this, _Val);
			return *this;
		}

		template <activation_fn_t _Fn>
		inline void activate() {
			cuda::activation_apply<_Fn>(*this);
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