#pragma once
#include "../vector.hpp"
#include "../vector_view.hpp"
#include "algebra/matrix_add_vector.cuh"
#include "algebra/matrix_sub_vector.cuh"
#include "algebra/matrix_add_scalar.cuh"
#include "algebra/matrix_mul_scalar.cuh"

namespace cuda {
	
	template <class _Ty, bool _Tr = false>
	class vector
		: public base::vector<_Ty, allocator<_Ty>, _Tr>
	{	
	protected:
		using _Mybase = base::vector<_Ty, allocator<_Ty>, _Tr>;
		using _Mybase::_Data;

	public:
		using _Mybase::_Mybase;

		template <bool _T>
		inline vector& operator+=(const vector<_Ty, _T>& _Other) {
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			cuda::matrix_add_vector(*this, _Other.T(), *this);
			return *this;
		}

		template <bool _T>
		inline vector& operator-=(const vector<_Ty, _T>& _Other) {
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			cuda::matrix_sub_vector(*this, _Other.T(), *this);
			return *this;
		}

		template <bool _T>
		inline vector operator+(const vector<_Ty, _T>& _Other) const {
			static_assert(!(_T && _Tr), "Cannot add transposed vectors");
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			vector _Result(_Mybase::size());
			cuda::matrix_add_vector(*this, _Other.T(), _Result);
			return _Result;
		}

		template <bool _T>
		inline vector operator-(const vector<_Ty, _T>& _Other) const {
			static_assert(!(_T && _Tr), "Cannot sub transposed vectors");
#ifdef DEBUG
			assert(_Mybase::rows() == _Other.rows());
#endif
			vector _Result(_Mybase::size());
			cuda::matrix_sub_vector(*this, _Other.T(), _Result);
			return _Result;
		}
		
		inline vector& operator+=(const _Ty& _Scalar) {
			cuda::matrix_add_scalar(*this, *this, _Scalar);
			return *this;
		}

		inline vector& operator-=(const _Ty& _Scalar) {
			cuda::matrix_add_scalar(*this, *this, -_Scalar);
			return *this;
		}

		inline vector& operator*=(const _Ty& _Scalar) {
			cuda::matrix_mul_scalar(*this, *this, _Scalar);
			return *this;
		}

		inline vector operator+(const _Ty& _Scalar) const {
			vector _Result(_Mybase::shape());
			cuda::matrix_add_scalar(*this, _Result, _Scalar);
			return _Result;
		}

		inline vector operator-(const _Ty& _Scalar) const {
			vector _Result(_Mybase::shape());
			cuda::matrix_add_scalar(*this, _Result, -_Scalar);
			return _Result;
		}

		inline vector operator*(const _Ty& _Scalar) const {
			vector _Result(_Mybase::shape());
			cuda::matrix_mul_scalar(*this, _Result, _Scalar);
			return _Result;
		}

		inline friend std::ostream& operator<<(std::ostream& _Os, const vector& _V) {
			if constexpr (!_Tr) {
				_Os << "[CUDA]\n[ " << _V.rows() << " ] (rows) [\n";
				for (size_t i = 0; i < _V.rows(); ++i) {
					_Os << "    " << cuda::from_cuda(&_V._Data[i]) << '\n';
				}
				_Os << "]" << std::endl;
			}
			else {
				_Os << "[CUDA]\n[ " << _V.cols() << " ] (cols) [\n    ";
				for (size_t i = 0; i < _V.cols(); ++i) {
					_Os << cuda::from_cuda(&_V._Data[i]) << ' ';
				}
				_Os << "\n]" << std::endl;
			}
			return _Os;
		}
	};
}