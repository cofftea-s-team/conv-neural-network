#pragma once
#include "../vector.hpp"
#include "../vector_view.hpp"
#include "algebra/matrix_add.hpp"
#include "algebra/matrix_add_vector.hpp"
#include "algebra/matrix_scalar_mul.hpp"

namespace host {
	
	template <class _Ty>
	class vector
		: public base::vector<_Ty, allocator<_Ty>, false>
	{
		using _Mybase = base::vector<_Ty, allocator<_Ty>, false>;
		using _Mybase::_Data;
		
	public:
		using _Mybase::_Mybase;

		inline auto T() {
			return base::transposed(*this);
		}

		inline vector& operator+=(const vector& _Other) {
			algebra::matrix_add_vector(*this, _Other, *this);
			return *this;
		}

		inline friend std::ostream& operator<<(std::ostream& _Os, const vector& _V) {
			_Os << "[HOST]\n[ " << _V.rows() << " ] (rows) [\n";
			for (size_t i = 0; i < _V.rows(); ++i) {
				_Os << "    " << _V._Data[i] << '\n';
			}
			_Os << "\n]" << std::endl;
			return _Os;
		}
	};
}