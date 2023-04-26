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

		template <bool _T2>
		inline auto& operator+=(const base::matrix<_Ty, _Alloc, _T2>& _Other) {
			static_assert(std::_Always_false<matrix_view>, "Cannot add to matrix_view (not implemented)");
			return *this;
		}

		template <bool _T2>
		inline auto& operator-=(const base::matrix<_Ty, _Alloc, _T2>& _Other) {
			static_assert(std::_Always_false<matrix_view>, "Cannot substract to matrix_view (not implemented)");
			return *this;
		}

		template <bool _T2>
		inline auto& operator*=(const base::matrix<_Ty, _Alloc, _T2>& _Other) {
			static_assert(std::_Always_false<matrix_view>, "Cannot mul with matrix_view (not implemented)");
			return *this;
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