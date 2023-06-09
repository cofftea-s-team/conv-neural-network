#pragma once
#include "../base/dual_matrix.hpp"

namespace cuda {
	template <class, bool>
	class matrix;

	template <class _Ty>
	using dual_matrix = base::dual_matrix<_Ty, matrix>;
}