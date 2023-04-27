#pragma once
#include "../base/vector.hpp"
#include "../base/vector_view.hpp"
#include "algebra/matrix_add.hpp"
#include "algebra/matrix_add_vector.hpp"
#include "algebra/matrix_scalar_mul.hpp"

namespace host {
	
	template <class _Ty, bool _Tr = false>
	class vector
		: public base::vector<_Ty, allocator<_Ty>, _Tr>
	{
	protected:
		using _Mybase = base::vector<_Ty, allocator<_Ty>, _Tr>;
		using _Mybase::_Data;
		
	public:
		using _Mybase::_Mybase;

		inline auto T() {
			return base::transposed(*this);
		}

		inline vector& operator+=(const vector& _Other) {
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			host::matrix_add_vector<std::plus<_Ty>>(*this, _Other, *this);
			return *this;
		}

		inline vector& operator-=(const vector& _Other) {
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			host::matrix_add_vector<std::minus<_Ty>>(*this, _Other, *this);
			return *this;
		}

		inline vector operator+(const vector& _Other) const {
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			vector _Result(_Mybase::size());
			host::matrix_add_vector<std::plus<_Ty>>(*this, _Other, _Result);
			return _Result;
		}
		
		inline vector operator-(const vector& _Other) const {
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			vector _Result(_Mybase::size());
			host::matrix_add_vector<std::minus<_Ty>>(*this, _Other, _Result);
			return _Result;
		}

		inline vector& operator+=(const _Ty& _Scalar) {
			host::matrix_add_scalar(*this, *this, _Scalar);
			return *this;
		}

		inline vector& operator-=(const _Ty& _Scalar) {
			host::matrix_add_scalar(*this, *this, -_Scalar);
			return *this;
		}

		inline vector& operator*=(const _Ty& _Scalar) {
			host::matrix_mul_scalar(*this, *this, _Scalar);
			return *this;
		}
		
		inline vector operator+(const _Ty& _Scalar) {
			vector _Result(_Mybase::shape());
			host::matrix_add_scalar(*this, _Result, _Scalar);
			return _Result;
		}
		
		inline vector operator-(const _Ty& _Scalar) {
			vector _Result(_Mybase::shape());
			host::matrix_add_scalar(*this, _Result, -_Scalar);
			return _Result;
		}

		inline vector operator*(const _Ty& _Scalar) {
			vector _Result(_Mybase::shape());
			host::matrix_mul_scalar(*this, _Result, _Scalar);
			return _Result;
		}

		inline friend std::ostream& operator<<(std::ostream& _Os, const vector& _V) {
			if constexpr (!_Tr) {
				_Os << "[HOST]\n[ " << _V.rows() << " ] (rows) [\n";
				for (size_t i = 0; i < _V.rows(); ++i) {
					_Os << "    " << _V._Data[i] << '\n';
				}
				_Os << "]" << std::endl;
			}
			else {
				_Os << "[HOST]\n[ " << _V.cols() << " ] (cols) [\n    ";
				for (size_t i = 0; i < _V.cols(); ++i) {
					_Os << _V._Data[i] << ' ';
				}
				_Os << "\n]" << std::endl;
			}
			return _Os;
		}
	};
}