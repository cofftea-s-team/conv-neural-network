#pragma once

#include "cuda/utils.hpp"
#include "cuda/matrix.hpp"

#include "host/utils.hpp"
#include "host/matrix.hpp"

#include "matrix_view.hpp"
#include "matrix.hpp"

template <class _Ty, base::allocator_t _Alloc>
inline constexpr void fill_normal_distribution(base::matrix<_Ty, _Alloc, false>& _Mat) {
	if constexpr (_Alloc::is_cuda()) {
		host::matrix<_Ty> _M_tmp(_Mat.rows(), _Mat.cols());
		host::fill_normal_distribution(_M_tmp);
		_Mat = _M_tmp;
	}
	else {
		host::fill_normal_distribution(_Mat);
	}
}

namespace cuda {

	
}
