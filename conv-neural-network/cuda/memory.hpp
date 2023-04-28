#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <exception>
#include <iostream>

namespace cuda {
	namespace details {
		template <class _Ty, size_t _Size>
		void* const _Alloc() {
			void* _Data;
			cudaMalloc(&_Data, _Size * sizeof(_Ty));
			return _Data;
		}

		constexpr size_t max_size = 8144 * 8144;
		void* const _Data = _Alloc<void*, max_size>();
	}
	
	template <class _Ty>
	inline _Ty* const get_memory(size_t N) {
		if (N * sizeof(_Ty) > details::max_size * sizeof(void*)) {
			std::cout << "not enought preallocated memory! (" << N * sizeof(_Ty) << " required, " << details::max_size * sizeof(void*) << " available)" << std::endl;
			throw std::exception("N is too big");
		}
		return reinterpret_cast<_Ty*>(details::_Data);
	}

}