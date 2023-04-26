#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils_cuda.cuh"

namespace cuda {
	namespace details {
		 
		constexpr size_t max_size = 8144 * 8144;
		static void* _Data;
	}
	
	template <class _Ty>
	inline _Ty* const get_memory(size_t N) {
		if (N > details::max_size) {
			throw std::bad_alloc();
		}

		return reinterpret_cast<_Ty*>(details::_Data);
	}

}