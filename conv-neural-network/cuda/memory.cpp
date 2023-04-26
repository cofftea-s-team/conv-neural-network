#include "memory.hpp"

void* _Data = cuda::alloc<void*>(cuda::details::max_size);

namespace cuda {
	template <class _Ty>
	inline _Ty* const get_memory(size_t N) {
		if (N * sizeof(_Ty) > details::max_size * sizeof(double)) {
			throw std::runtime_error("N is too big");
		}
		return reinterpret_cast<_Ty*>(_Data);
	}
	template float* const get_memory<float>(size_t);
	template double* const get_memory<double>(size_t);
	template bfloat16* const get_memory<bfloat16>(size_t);
}
