#pragma once
#include "utils.hpp"
#include "vector.hpp"

namespace cuda {
	template <class, bool>
	class vector;
}

namespace host {
	template <class, bool>
	class vector;
}

namespace base {
	template <class _Ty, allocator_t _Alloc, bool _T = true>
	class vector_view
		: public std::conditional_t<_Alloc::is_cuda(), cuda::vector<_Ty, _T>, host::vector<_Ty, _T>>
	{
	protected:
		using _Mybase = std::conditional_t<_Alloc::is_cuda(), cuda::vector<_Ty, _T>, host::vector<_Ty, _T>>;
		using _Mybase::_Data;
		using _Mybase::_Rows;
		using _Mybase::_Cols;
		using _Mybase::_Is_owner;
		using _Mybase::_Al;
	public:

		template <class _Ty2, allocator_t _Alloc2, bool _T2>
		friend class vector_view;

		inline vector_view(vector<_Ty, _Alloc, !_T>& _Vec) {
			_Data = _Vec.data();
			_Cols = _Vec.rows();
			_Rows = _Vec.cols();
			_Is_owner = false;
		}

		inline vector_view(const vector<std::remove_const_t<_Ty>, _Alloc, !_T>& _Vec) {
			_Data = _Vec.data();
			_Cols = _Vec.rows();
			_Rows = _Vec.cols();
			_Is_owner = false;
		}

		inline vector_view(size_t _N, size_t _M, _Ty* _Ptr) {
			_Is_owner = false;
			_Data = _Ptr;
			_Cols = _N;
			_Rows = _M;
		}

		inline vector_view(vector_view&& _Other) noexcept {
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Data = _Other._Data;
			_Is_owner = _Other._Is_owner;
			_Other._Is_owner = false;
			_Other._Data = nullptr;
		}

		inline auto T() {
			return vector_view<_Ty, _Alloc, !_T>(*this);
		}

		inline friend ostream& operator<<(ostream& _Os, const vector_view& _M) {
			if constexpr (_Al.is_cuda())
				_Os << "[CUDA]";
			else
				_Os << "[HOST]";
				
			_Os << "\n[ " << _M.size() << " ] (";
			if constexpr (_M.is_transposed())
				_Os << "cols";
			else 
				_Os << "rows";

			_Os << ") [\n";
			if constexpr (_M.is_transposed()) _Os << "    ";
			const _Ty* _Ptr = _M.data();
			for (int i = 0; i < _M.size(); ++i) {
				if constexpr (!_M.is_transposed())
					_Os << "    ";
				
				if constexpr (_Al.is_cuda())
					_Os << cuda::from_cuda(&_Ptr[i]);
				else
					_Os << _Ptr[i];

				if constexpr (_M.is_transposed())
					_Os << ' ';
				else
					_Os << '\n';
			}
			if constexpr (_M.is_transposed()) _Os << '\n';
			_Os << "]" << endl;

			return _Os;
		}
	};

	template <class _Ty, allocator_t _Alloc, bool _T>
	inline auto transposed(vector<_Ty, _Alloc, _T>& _V) {
		return vector_view<_Ty, _Alloc, !_T>(_V);
	}

	template <class _Ty, allocator_t _Alloc, bool _T>
	inline auto create_view(vector<_Ty, _Alloc, _T>& _V) {
		return vector_view<_Ty, _Alloc, _T>(1, _V.rows(), _V.data());
	}
	
}