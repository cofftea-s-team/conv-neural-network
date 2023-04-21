#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils_cuda.cuh"

namespace cuda {
	namespace details {
		
		constexpr size_t max_size = 8144 * 8144;
		static void* const _Data = cuda::alloc<void*>(max_size);
	}
	
	template <class _Ty>
	inline _Ty* const get_memory(size_t N) {
		if (N * sizeof(_Ty) > details::max_size * sizeof(double)) {
			throw std::runtime_error("N is too big");
		}
		return reinterpret_cast<_Ty*>(details::_Data);
	}

}