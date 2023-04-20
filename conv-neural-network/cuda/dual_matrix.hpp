#pragma once
#include "../dual_matrix.hpp"

namespace cuda {
	template <class _Ty>
	using dual_matrix = base::dual_matrix<_Ty, matrix>;
}