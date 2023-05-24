#include "range_max.cuh"

#define BLOCK_DIM 1024
#define SHMEM_SIZE BLOCK_DIM * 4

namespace cuda {

	template <class _Ty>
	__device__ inline const _Ty& max(const _Ty& _Left, const _Ty& _Right) {
		return _Left > _Right ? _Left : _Right;
	}

	template <class _Ty>
	__global__ void max_kernel(const _Ty* _Data, _Ty* _Tmp, size_t N) {

		__shared__ _Ty _Partial_sum[SHMEM_SIZE];

		size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

		if (i >= N) {
			_Partial_sum[threadIdx.x] = -1e10;
		}
		else {
			if (i + blockDim.x < N)
				_Partial_sum[threadIdx.x] = cuda::max(_Data[i], _Data[i + blockDim.x]);
			else
				_Partial_sum[threadIdx.x] = _Data[i];
		}

		__syncthreads();

		for (size_t j = blockDim.x / 2; j > 0; j >>= 1) {
			if (threadIdx.x < j) {
				_Partial_sum[threadIdx.x] = cuda::max(_Partial_sum[threadIdx.x], _Partial_sum[threadIdx.x + j]);
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			_Tmp[blockIdx.x] = _Partial_sum[0];
		}
	}

	template <class _Ty>
	_Ty _range_max(const _Ty* _Data, size_t N) {
		_Ty* _Tmp_res = get_memory<_Ty>(N * 2);

		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((N + threads.x - 1) / threads.x);

		max_kernel<<<blocks, threads>>>(_Data, _Tmp_res, N);
		max_kernel<<<1, threads>>>(_Tmp_res, _Tmp_res, N);

		_Ty _Res;
		cuda::memcpy(_Tmp_res, &_Res, 1, DeviceToHost);

		return _Res;
	}

#define INSTANTIATE(_Ty) \
	template _Ty _range_max(const _Ty* _Data, size_t N);

	INSTANTIATE(float);
	INSTANTIATE(double);
	INSTANTIATE(bfloat16);
}