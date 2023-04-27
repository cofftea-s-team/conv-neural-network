#include "../base/dual_matrix.hpp"

namespace host {
	template <class _Ty>
	using dual_matrix = base::dual_matrix<_Ty, matrix>;
}