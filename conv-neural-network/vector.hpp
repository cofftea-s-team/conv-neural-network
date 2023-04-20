#pragma once
#include "matrix.hpp"

namespace base {
	template <class, allocator_t, bool>
	class vector_view;

	template <class _Ty, allocator_t _All, bool _T>
	class vector
		: public matrix<_Ty, _All, _T>
	{
	protected:
		using _Mybase = matrix<_Ty, _All, _T>;
		using _Mybase::_Data;
		using _Mybase::_Rows;

		inline vector() = default;
	
	public:

		inline auto T() {
			return vector_view<_Ty, _All, !_T>(*this);
		}

		inline vector(size_t _Rows)
			: _Mybase(_Rows, 1)
		{ }

		template <allocator_t _Other_all>
		inline vector(const vector<_Ty, _Other_all, _T>& _Other)
			: _Mybase(_Other)
		{ }
	};
}