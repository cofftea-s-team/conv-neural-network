#pragma once
#include <execution>
#include <array>

namespace host::algebra {
	namespace parallel {
		namespace details {
			template <size_t N>
			inline constexpr auto range() {
				std::array<uint32_t, N> _Array;
				std::iota(_Array.begin(), _Array.end(), 0);
				return _Array;
			}

			inline constexpr size_t max_indice = 16384;
			inline static constexpr auto indices = range<max_indice>();
		}

		template <class _Fn>
		inline constexpr void parallel_for(size_t _Count, _Fn _Func) {
			std::for_each(std::execution::par,
				details::indices.begin(), std::next(details::indices.begin(), _Count), _Func);
		}
	}
}