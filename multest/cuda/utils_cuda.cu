#include "utils_cuda.cuh"

namespace cuda {
#define BLOCK_SIZE 1024

	template <RAND_MODE _Mode, class _Ty>
	__global__ void random_kernel(_Ty* _Data, size_t N, curandState* _State) {
		int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if (i < N) {
			size_t _State_idx = i & 1023;
			curand_init(1337, _State_idx, 0, &_State[_State_idx]);
			
			if constexpr (_Mode == UNIFORM) {
				_Data[i] = curand_uniform(&_State[_State_idx]);
			}
			else if constexpr (_Mode == NORMAL) {
				_Data[i] = curand_normal(&_State[_State_idx]);
			}
		}
	}

	template <RAND_MODE _Mode, class _Ty>
	inline void _fill_random(_Ty* _Data, size_t N) {
		curandState* _State;
		cudaMalloc(&_State, 1024 * sizeof(curandState));
		
		dim3 blockDim(BLOCK_SIZE);
		dim3 gridDim((N - 1) / BLOCK_SIZE + 1);
		
		random_kernel<_Mode>
			<<<gridDim, blockDim>>>(_Data, N, _State);

		cudaFree(_State);
	}

	template void _fill_random<NORMAL, double>(double*, size_t);
	template void _fill_random<UNIFORM, double>(double*, size_t);
	
	template void _fill_random<NORMAL, float>(float*, size_t);
	template void _fill_random<UNIFORM, float>(float*, size_t);

	template void _fill_random<NORMAL, bfloat16>(bfloat16*, size_t);
	template void _fill_random<UNIFORM, bfloat16>(bfloat16*, size_t);
}
