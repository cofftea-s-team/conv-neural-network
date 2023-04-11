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
		cuda::fill_random<NORMAL>(_Mat);
	}
	else {
		host::fill_normal_distribution(_Mat);
	}
}

template <class _Mat1, class _Mat2>
inline base::shape get_mul_shape(const _Mat1& _Left, const _Mat2& _Right) {
	return base::shape(_Left.rows(), _Right.cols());
}

namespace cuda {

	
}
