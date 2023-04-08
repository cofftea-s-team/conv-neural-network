#include "kernel.cuh"
#include "cuda_mul_shared.cuh"
#include "matrix.hpp"

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
	class matrix 
		: public base::matrix<_Ty, cuda::allocator<_Ty>, false> 
	{
	protected:
		using _Mybase = base::matrix<_Ty, cuda::allocator<_Ty>, false>;
		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Data;
	public:
		using _Mybase::_Mybase;
		
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

	};
}