#include "kernel.cuh"
#include "matrix.hpp"

#include <iostream>

namespace host {
	using std::cout;
	using std::endl;
	using std::ostream;
	template <class _Ty>
	struct allocator 
		: public base::allocator<_Ty, false> 
	{
		constexpr _Ty* alloc(size_t _Count) const override  {
			return cuda::alloc_paged<_Ty>(_Count);
		}

		constexpr void free(_Ty* ptr) const override {
			cuda::free_paged<_Ty>(ptr);
		}
	};

	template <class _Ty>
	class matrix 
		: public base::matrix<_Ty, allocator<_Ty>, false> 
	{
		using _Mybase = base::matrix<_Ty, allocator<_Ty>, false>;
	public:
		using _Mybase::_Mybase;

		template <base::allocator_t _Other_all, bool _T2>
		inline matrix& operator=(const base::matrix<_Ty, _Other_all, _T2>& _Other) {
			_Mybase::operator=(_Other);
			return *this;
		}

		inline friend ostream& operator<<(ostream& _Os, const matrix& _M) {
			cout << "[HOST]\n[" << _M.rows() << "x" << _M.cols() << "] (rows x cols) {\n";
			const _Ty* _Ptr = _M.data();
			for (int i = 0; i < _M.rows(); ++i) {
				cout << "    ";
				for (int j = 0; j < _M.cols(); ++j) {
					_Os << _Ptr[i * _M.cols() + j] << " ";
				}
				_Os << endl;
			}
			cout << "}" << endl;
			
			return _Os;
		}
	};
}