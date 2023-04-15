#pragma once
#include "utils.hpp"
#include "../vector.hpp"

namespace cuda {
	template <class _Ty>
	class vector
		: public base::vector<_Ty, allocator<_Ty>, false>
	{
		using _Mybase = base::vector<_Ty, allocator<_Ty>, false>;
		using _Mybase::_Data;
	public:
		using _Mybase::_Mybase;

		inline friend std::ostream& operator<<(std::ostream& _Os, const vector& _V) {
			_Os << "[CUDA]\n[ " << _V.rows() << " ] (rows) [\n";
			for (size_t i = 0; i < _V.rows(); ++i) {
				_Os << "    " << cuda::from_cuda(&_V._Data[i]) << '\n';
			}
			_Os << "\n]" << std::endl;
			return _Os;
		}
	};
}