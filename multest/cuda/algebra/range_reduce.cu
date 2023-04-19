#include "range_reduce.cuh"

#define BLOCK_DIM 1024
#define SHMEM_SIZE BLOCK_DIM * 4

namespace cuda {
	
	template <class _Ty>
	__global__ void sum_reduction(const _Ty* _Data, _Ty* _Tmp, size_t N) {
		
		__shared__ _Ty _Partial_sum[SHMEM_SIZE];
		
		size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

		if (i >= N) {
			return;
		}
		
		_Partial_sum[threadIdx.x] = _Data[i] + _Data[i + blockDim.x];

		__syncthreads();

		for (size_t j = blockDim.x / 2; j > 0; j >>= 1) {
			if (threadIdx.x < j) {
				_Partial_sum[threadIdx.x] = _Partial_sum[threadIdx.x] + _Partial_sum[threadIdx.x + j];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			_Tmp[blockIdx.x] = _Partial_sum[0];
		}
	}

	template <class _Ty>
	_Ty _range_reduce(const _Ty* _Data, size_t N, size_t M) {
		_Ty* _Tmp_res = get_memory<_Ty>(N * M * 2);

		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((N * M + threads.x - 1) / threads.x);

		sum_reduction<<<blocks, threads >>>(_Data, _Tmp_res, N * M);
		sum_reduction<<<1, threads>>>(_Tmp_res, _Tmp_res, N * M );
		
		_Ty _Res;
		cuda::memcpy(_Tmp_res, &_Res, 1, DeviceToHost);

		return _Res;
	}

	template double _range_reduce<double>(const double*, size_t, size_t);
	template float _range_reduce<float>(const float*, size_t, size_t);
	template bfloat16 _range_reduce<bfloat16>(const bfloat16*, size_t, size_t);
}