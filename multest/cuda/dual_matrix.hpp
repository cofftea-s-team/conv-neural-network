#include "matrix.hpp"
#include "../matrix_view.hpp"

namespace cuda {
	template <class _Ty>
	class dual_matrix
		: public matrix<_Ty>
	{
		using _Mybase = matrix<_Ty>;

		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Data;
		
		using _Mybase::_Al;
	public:
		using _Mybase::_Mybase;
		
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
		inline auto mul(const base::matrix<_Ty, cuda::allocator<_Ty>, _T>& _Other) {
			auto view = base::matrix_view<_Ty, cuda::allocator<_Ty>, false>(_Other.cols(), _Rows, _Mul_result);
			matrix_multiply(*this, _Other, view);
			return view;
		}
	private:
		_Ty* _Mul_result = nullptr;
	};
}