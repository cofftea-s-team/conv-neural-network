#pragma once
#include "../matrix.hpp"
#include "../cuda/utils.hpp"
#include "../matrix_view.hpp"
#include "utils.hpp"
#include <iostream>

namespace cuda {
	template <class _Ty>
	class matrix;
}

namespace host {
	using std::cout;
	using std::endl;
	using std::ostream;

	template <class _Ty>
	class matrix 
		: public base::matrix<_Ty, allocator<_Ty>, false> 
	{
		using _Mybase = base::matrix<_Ty, allocator<_Ty>, false>;
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
			if constexpr (std::is_same_v<_Ty, float>) {
				assert(_Rows % 16 == 0 && _Cols % 16 == 0);
			}
		}

		template <base::allocator_t _Other_all, bool _T2>
		inline matrix& operator=(const base::matrix<_Ty, _Other_all, _T2>& _Other) {
			_Mybase::operator=(_Other);
			return *this;
		}

		template <bool _T>
		inline matrix mul(const base::matrix<_Ty, allocator<_Ty>, _T>& _Other) const {
#ifdef DEBUG
			assert(_Cols == _Other.rows());
#endif // !DEBUG
			matrix _Res(_Rows, _Other.cols());
			host::matrix_multiply(*this, _Other, _Res);
			return _Res;
		}

		template <bool _T>
		inline matrix& operator+=(const base::vector<_Ty, host::allocator<_Ty>, _T>& _Other) {
			host::matrix_add_vector(*this, _Other, *this);
			return *this;
		}

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

		template <activation_fn_t _Fn>
		inline void activate() {
			host::activation_apply<_Fn>(*this);
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