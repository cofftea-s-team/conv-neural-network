#include "utils_cuda.cuh"

#define BLOCK_SIZE 512
namespace cuda {

	__global__ void setup_curand_state_kernel(curandState* state, size_t N, size_t _Seed) {
		size_t i = threadIdx.x + blockDim.x * blockIdx.x;
		
		if (i < N) {
			curand_init(_Seed, i, 0, &state[i]);
		}
	}

	template <RAND_MODE _Mode, class _Ty>
	__global__ void random_kernel(_Ty* _Data, size_t N, curandState* _State) {
		size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if (i < N) {
			if constexpr (_Mode == UNIFORM) {
				_Data[i] = curand_uniform(&_State[i]) * 2 - 1;
			}
			else if constexpr (_Mode == NORMAL) {
				_Data[i] = curand_normal(&_State[i]) * 0.1;
			}
		}
	}

	template <RAND_MODE _Mode, class _Ty>
	inline void _fill_random(_Ty* _Data, size_t N) {
		auto _State = get_memory<curandState>(N);

		dim3 threads(BLOCK_SIZE);
		dim3 blocks((N - 1) / BLOCK_SIZE + 1);
		
		size_t _Seed = rand();

		setup_curand_state_kernel<<<blocks, threads>>>(_State, N, _Seed);
		
		random_kernel<_Mode>
			<<<blocks, threads>>>(_Data, N, _State);

	}

	template void _fill_random<NORMAL, double>(double*, size_t);
	template void _fill_random<UNIFORM, double>(double*, size_t);
	
	template void _fill_random<NORMAL, float>(float*, size_t);
	template void _fill_random<UNIFORM, float>(float*, size_t);

	template void _fill_random<NORMAL, bfloat16>(bfloat16*, size_t);
	template void _fill_random<UNIFORM, bfloat16>(bfloat16*, size_t);

}
