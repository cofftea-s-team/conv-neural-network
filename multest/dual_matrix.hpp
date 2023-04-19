#pragma once
#include "matrix_view.hpp"

namespace base {
	template <class _Ty, template <class _T> class _Base>
	class dual_matrix
		: public _Base<_Ty>
	{
		using _Mybase = _Base<_Ty>;
		
		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Data;

		using _Mybase::_Al;
	public:
		using alloc = typename _Base<_Ty>::alloc;
		using _Mybase::_Mybase;

		inline ~dual_matrix() {
			free_result();
		}

		template <base::allocator_t _Other_all, bool _T2>
		inline dual_matrix& operator=(const base::matrix<_Ty, _Other_all, _T2>& _Other) {
			_Mybase::operator=(_Other);
			return *this;
		}

		inline void alloc_result(size_t _Count) {
			_Mul_result = _Al.alloc(_Count);
		}

		inline void alloc_result(base::shape _Shape) {
			_Mul_result = _Al.alloc(_Shape.second * _Shape.first);
		}

		inline void free_result() {
			_Al.free(_Mul_result);
		}

		template <bool _T>
		inline auto mul(const base::matrix<_Ty, alloc, _T>& _Other) {
			auto view = base::matrix_view<_Ty, alloc, false>(_Other.cols(), _Rows, _Mul_result);
			matrix_multiply(*this, _Other, view);
			return view;
		}
	private:
		_Ty* _Mul_result = nullptr;
	};
}