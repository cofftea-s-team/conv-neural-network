#include "matrix_sum.cuh"

#define BLOCK_DIM 256
#define SHMEM_SIZE BLOCK_DIM * 4

namespace cuda {

	template <class _Ty>
	__global__ void column_sum_reduction(const _Ty* _Data, _Ty* _Tmp, size_t N, size_t M) {

		__shared__ _Ty _Partial_sum[SHMEM_SIZE];

		size_t i = (blockIdx.x * (blockDim.x) + threadIdx.x) * N;

		if (i < M * N) {
			_Partial_sum[threadIdx.x] = _Data[i];//; +_Data[i + blockDim.x * N];
		}
		else {
			_Partial_sum[threadIdx.x] = 0.f;
		}


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
	__global__ void sum_reduction(const _Ty* _Data, _Ty* _Tmp, size_t N) {

		__shared__ _Ty _Partial_sum[SHMEM_SIZE];

		size_t i = blockIdx.x * (blockDim.x) + threadIdx.x;

		if (i >= N) {
			_Partial_sum[threadIdx.x] = 0.f;
		}
		else {
			_Partial_sum[threadIdx.x] = _Data[i];
		}

		_Partial_sum[threadIdx.x] = _Data[i];// +_Data[i + blockDim.x];

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

	// causes problems
    template <class _Ty>
    _Ty* reduce_column(const _Ty* _Data, size_t N, size_t M) {
		_Ty* _Tmp_res = get_memory<_Ty>(N * M * 2);

		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((M + BLOCK_DIM - 1) / BLOCK_DIM);

		column_sum_reduction<<<blocks, threads>>>(_Data, _Tmp_res, N, M);
		sum_reduction<<<1, threads>>>(_Tmp_res, _Tmp_res, M);

		return _Tmp_res;
    }

	template <bool _T1, bool _T2, class _Ty>
	void _matrix_sum0(const _Ty* A, _Ty* V, size_t N, size_t M) {
		for (size_t i = 0; i < M; ++i) {
			cuda::memcpy(cuda::_range_reduce(&A[i * N], N, 1), &V[i], 1, DeviceToDevice);
		}
	}

	template <bool _T1, bool _T2, class _Ty>
	void _matrix_sum1(const _Ty* A, _Ty* V, size_t N, size_t M) {
		for (size_t i = 0; i < N; ++i) {
			cuda::memcpy(reduce_column(&A[i], N, M), &V[i], 1, DeviceToDevice);
		}
	}

#define INSTANTIATE(_Ty) \
	template void _matrix_sum0<false, false, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum0<false, true, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum0<true, false, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum0<true, true, _Ty>(const _Ty*, _Ty*, size_t, size_t);  \
	template void _matrix_sum1<false, false, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum1<false, true, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum1<true, false, _Ty>(const _Ty*, _Ty*, size_t, size_t); \
	template void _matrix_sum1<true, true, _Ty>(const _Ty*, _Ty*, size_t, size_t);

	INSTANTIATE(double);
	INSTANTIATE(float);
	//INSTANTIATE(bfloat16);
}


