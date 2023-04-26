#pragma once

#include "cuda/utils.hpp"
#include "cuda/matrix.hpp"

#include "host/utils.hpp"
#include "host/matrix.hpp"

#include "matrix_view.hpp"
#include "matrix.hpp"
#include <tuple>

namespace utils {

	template <class _Ty, base::allocator_t _Alloc, bool _T>
	inline constexpr void generate_normal(base::matrix<_Ty, _Alloc, _T>& _Mat) {
		if constexpr (_Alloc::is_cuda()) {
			cuda::fill_random<NORMAL>(_Mat);
		}
		else {
			host::fill_normal_distribution(_Mat);
		}
	}
	
	template <class _Ty, base::allocator_t _Alloc, bool _T>
	inline constexpr void generate_uniform(base::matrix<_Ty, _Alloc, _T>& _Mat) {
		if constexpr (_Alloc::is_cuda()) {
			cuda::fill_random<UNIFORM>(_Mat);
		}
		else {
			host::fill_uniform_distribution(_Mat);
		}
	}

	template <class _Mat1, class _Mat2>
	inline base::shape get_mul_shape(const _Mat1& _Left, const _Mat2& _Right) {
		return base::shape(_Left.rows(), _Right.cols());
	}

	inline constexpr base::shape mul(base::shape _Left, base::shape _Right) {
		return base::shape(_Left.rows(), _Right.cols());
	}

	template<class _Lambda, size_t I = 0, class... _TArgs>
	inline constexpr std::enable_if_t<I == sizeof...(_TArgs), void> for_each(std::tuple<_TArgs...>& _T, _Lambda _Fn)
	{ }

	template<class _Lambda, size_t I = 0, class... _TArgs>
	inline constexpr std::enable_if_t<I < sizeof...(_TArgs), void> for_each(std::tuple<_TArgs...>& _T, _Lambda _Fn) {
		_Fn(std::get<I>(_T));
		utils::for_each<_Lambda, I + 1, _TArgs...>(_T, _Fn);
	}

	template<int N, class _Lambda, int I = N - 1, class... _TArgs>
	inline constexpr std::enable_if_t<I == -1, void> rfor_each(std::tuple<_TArgs...>& _T, _Lambda _Fn)
	{ }

	template<int N, class _Lambda, int I = N - 1, class... _TArgs>
	inline constexpr std::enable_if_t<I >= 0, void> rfor_each(std::tuple<_TArgs...>& _T, _Lambda _Fn) {
		_Fn(std::get<I>(_T));
		utils::rfor_each<N, _Lambda, I - 1, _TArgs...>(_T, _Fn);
	}
}