#pragma once
#include "matrix.hpp"

namespace cuda {
	template <class _Ty>
	class vector
		: public matrix<_Ty>
	{

	};
}