#pragma once
#include "utils.hpp"
#include "algebra/matrix_mul.cuh"
#include "algebra/matrix_add.cuh"
#include "algebra/matrix_sub.cuh"
#include "activations/activation.cuh"
#include "../activations.hpp"
#include "../matrix.hpp"
#include "../types.hpp"

namespace host {
	template <class _Ty>
	class matrix;
}

namespace cuda {
	
	using std::cout;
	using std::endl;
	using std::ostream;
	template <class _Ty>
	struct allocator
		: public base::allocator<_Ty, true> 
	{
		constexpr _Ty* alloc(size_t _Count) const override {
			return cuda::alloc<_Ty>(_Count);
		}

		constexpr void free(_Ty* ptr) const override {
			cuda::free<_Ty>(ptr);
		}
	};

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
			_STL_ASSERT(_Mybase::cols() == _Other.rows(), "matrix multiplication: cols != rows");
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
			_STL_ASSERT(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows(), "matrix addition: cols != cols || rows != rows");
#endif // DEBUG
			cuda::matrix_add(*this, _Other, *this);
			return *this;
		}

		template <bool _T>
		inline matrix& operator-=(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
#ifdef DEBUG
			_STL_ASSERT(_Mybase::cols() == _Other.cols() && _Mybase::rows() == _Other.rows(), "matrix subtraction: cols != cols || rows != rows");
#endif // DEBUG
			cuda::matrix_sub(*this, _Other, *this);
			return *this;
		}

		template <activation_fn_t _Fn>
		inline void activate() {
			cuda::activation_apply<_Fn>(*this);
		}

		inline friend ostream& operator<<(ostream& _Os, const matrix& _M) {
			cout << "[CUDA]\n[" << _M.rows() << "x" << _M.cols() << "] (rows x cols) {\n";
			const _Ty* _Ptr = _M.data();
			for (int i = 0; i < _M.rows(); ++i) {
				cout << "    ";
				for (int j = 0; j < _M.cols(); ++j) {
					_Os << cuda::from_cuda(&_Ptr[i * _M.cols() + j]) << " ";
				}
				_Os << endl;
			}
			cout << "}" << endl;

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

	template <class _Ty>
	struct cuda_to_host_matrix_iterator {
		struct element {
			inline element(_Ty* _Ptr)
				: _Ptr(_Ptr)
			{ }

			inline element& operator=(const _Ty& _Val) {
				cuda::to_cuda(&_Val, _Ptr);
				return *this;
			}

			inline operator _Ty() const {
				return cuda::from_cuda(_Ptr);
			}

			inline friend ostream& operator<<(ostream& _Os, const element& _El) {
				_Os << cuda::from_cuda(_El._Ptr);
				return _Os;
			}
			
			_Ty* _Ptr;
		};
		inline cuda_to_host_matrix_iterator(_Ty* _Ptr) 
			: _Ptr(_Ptr) 
		{}
		
		inline cuda_to_host_matrix_iterator& operator++() {
			++_Ptr;
			return *this;
		}
		
		inline cuda_to_host_matrix_iterator operator++(int) {
			cuda_to_host_matrix_iterator _Tmp = *this;
			++_Ptr;
			return _Tmp;
		}
		
		inline cuda_to_host_matrix_iterator& operator--() {
			--_Ptr;
			return *this;
		}

		inline cuda_to_host_matrix_iterator operator--(int) {
			cuda_to_host_matrix_iterator _Tmp = *this;
			--_Ptr;
			return _Tmp;
		}

		inline cuda_to_host_matrix_iterator& operator+=(int _Off) {
			_Ptr += _Off;
			return *this;
		}

		inline cuda_to_host_matrix_iterator& operator-=(int _Off) {
			_Ptr -= _Off;
			return *this;
		}

		inline cuda_to_host_matrix_iterator operator+(int _Off) const {
			return cuda_to_host_matrix_iterator(_Ptr + _Off);
		}

		inline cuda_to_host_matrix_iterator operator-(int _Off) const {
			return cuda_to_host_matrix_iterator(_Ptr - _Off);
		}

		inline int operator-(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr - _Other._Ptr;
		}

		inline bool operator==(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr == _Other._Ptr;
		}
		
		inline bool operator!=(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr != _Other._Ptr;
		}

		inline bool operator<(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr < _Other._Ptr;
		}

		inline bool operator>(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr > _Other._Ptr;
		}

		inline bool operator<=(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr <= _Other._Ptr;
		}

		inline bool operator>=(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr >= _Other._Ptr;
		}

		inline element operator*() const {
			return element(_Ptr);
		}
	private:
		_Ty* _Ptr;
	};
}