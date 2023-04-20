#pragma once
#include "matrix.hpp"

namespace cuda {
	template <class _Ty, bool _T>
	class matrix;
}

namespace host {
	template <class _Ty, bool _T>
	class matrix;
}

namespace base {
	using std::cout;
	using std::endl;
	using std::ostream;

	template <class _Ty, allocator_t _Alloc, bool _T = true>
	class matrix_view
		: public std::conditional_t<_Alloc::is_cuda(), cuda::matrix<_Ty, _T>, host::matrix<_Ty, _T>>
	{
		using _Mybase = std::conditional_t<_Alloc::is_cuda(), cuda::matrix<_Ty, _T>, host::matrix<_Ty, _T>>;
		

	public:
		template <class _Ty2, allocator_t _Alloc2, bool _T2>
		friend class matrix_view;
		
		using _Mybase::_Data;
		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Is_owner;
		using _Mybase::_Al;

		inline matrix_view(matrix<_Ty, _Alloc, !_T>& _Matrix) {
			_Data = _Matrix.data();
			_Cols = _Matrix.rows();
			_Rows = _Matrix.cols();
			_Is_owner = false;
		}
		
		inline matrix_view(size_t _N, size_t _M, _Ty* _Ptr) {
			_Is_owner = false;
			_Data = _Ptr;
			_Cols = _N;
			_Rows = _M;
		}

		inline matrix_view(matrix_view&& _Other) noexcept {
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Data = _Other._Data;
			_Is_owner = _Other._Is_owner;
			_Other._Is_owner = false;
			_Other._Data = nullptr;
		}

		inline auto T() {
			return matrix_view<_Ty, _Alloc, !_T>(*this);
		}
		
		inline friend ostream& operator<<(ostream& _Os, const matrix_view& _M) {
			if constexpr (_Al.is_cuda())
				_Os << "[CUDA]\n[" << _M.rows() << "x" << _M.cols() << "] (rows x cols) {\n";
			else
				_Os << "[HOST]\n[" << _M.rows() << "x" << _M.cols() << "] (rows x cols) {\n";
			
			const _Ty* _Ptr = _M.data();
			for (int j = 0; j < _M.rows(); ++j) {
				_Os << "    ";
				for (int i = 0; i < _M.cols(); ++i) {
					if constexpr (_Al.is_cuda())
						_Os << cuda::from_cuda(&_Ptr[i * _M.rows() + j]) << " ";
					else
						_Os << _Ptr[i * _M.rows() + j] << " ";
				}
				_Os << endl;
			}
			_Os << "}" << endl;

			return _Os;
		}
	};

	template <class _Ty, allocator_t _Alloc, bool _T>
	inline auto transposed(matrix<_Ty, _Alloc, _T>& _Matrix) {
		return matrix_view<_Ty, _Alloc, !_T>(_Matrix);
	}

	template <class _Ty, allocator_t _Alloc, bool _T>
	inline auto create_view(matrix<_Ty, _Alloc, _T>& _Matrix) {
		return matrix_view<_Ty, _Alloc, _T>(_Matrix);
	}
}