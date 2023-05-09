#include "dropout.cuh"

#define BLOCK_SIZE 512

namespace cuda {
	
	__global__ void setup_kernel(curandState* _State, size_t N, size_t _Seed) {
		size_t i = threadIdx.x + blockDim.x * blockIdx.x;
		
		if (i < N) {
			curand_init(_Seed, i, 0, &_State[i]);
		}
	}

	template <class _Ty>
	__global__ void apply_dropout_kernel(_Ty* _Data, size_t N, _Ty _Dropout_rate, curandState* _State) {
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
		if (i < N) {
			_Ty _Rnd = curand_uniform(&_State[i]);
			if (_Rnd < _Dropout_rate) {
				_Data[i] = 0.f;
			}
		}
	}


	template <class _Ty>
	void _apply_dropout(_Ty* _Data, size_t N, _Ty _Dropout_rate) {
		auto _State = get_memory<curandState>(N);
		
		dim3 threads(BLOCK_SIZE);
		dim3 blocks((N - 1) / BLOCK_SIZE + 1);

		int _Seed = rand();

		setup_kernel<<<blocks, threads>>>(_State, N, _Seed);

		apply_dropout_kernel<_Ty>
			<<<blocks, threads>>>(_Data, N, _Dropout_rate, _State);
	}

	template void _apply_dropout<double>(double*, size_t, double);
	template void _apply_dropout<float>(float*, size_t, float);
	template void _apply_dropout<bfloat16>(bfloat16*, size_t, bfloat16);
}