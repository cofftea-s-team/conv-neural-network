#include "activation.cuh"

#define BLOCK_DIM 1024

namespace cuda {
	
	template <class _Applier, class _Ty>
	__global__ void activation_apply_kernel(_Ty* _Data, size_t N) {
		const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < N) {
			_Data[i] = _Applier::apply(_Data[i]);
		}
	}

	template <class _Applier, class _Ty>
	__global__ void activation_apply_softmax_kernel(_Ty* _Data, size_t N, const _Ty* _Sum) {
		const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < N) {
			_Data[i] = _Applier::apply(_Data[i], *_Sum);
		}
	}

	template <class _Ty>
	__global__ void activate_softmax_kernel(_Ty* _Data, size_t N, size_t M) {
		const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= M) return;

		size_t _Row = i * N;

		_Ty _Max = _Data[_Row];
		for (size_t j = 1; j < N; ++j) {
			if (_Data[_Row + j] > _Max) {
				_Max = _Data[_Row + j];
			}
		}
		
		_Ty _Sum = 0.f;
		for (size_t j = 0; j < N; ++j) {
			_Data[_Row + j] = exp(_Data[_Row + j] - _Max);
			_Sum += _Data[_Row + j];
		}

		for (size_t j = 0; j < N; ++j) {
			_Data[_Row + j] /= _Sum;
		}
	}

	template <class _Activation_class, class _Ty>
	void _forward_apply(_Ty* _Data, size_t N, size_t M) {
		if constexpr (std::is_same_v<_Activation_class, cnn::softmax>) {
			//_softmax_apply(_Data, N, M);
			__softmax_apply(_Data, N, M);
		}
		else {
			using applier = forwarder<_Activation_class>;

			const dim3 threads(BLOCK_DIM);
			const dim3 blocks((N * M + BLOCK_DIM - 1) / BLOCK_DIM);

			activation_apply_kernel<applier>
				<<<blocks, threads>>>(_Data, N * M);
		}
	}

	template <class _Activation_class, class _Ty>
	void _backward_apply(_Ty* _Data, size_t N, size_t M) {
		using applier = backwarder<_Activation_class>;

		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((N * M + BLOCK_DIM - 1) / BLOCK_DIM);

		activation_apply_kernel<applier>
			<< <blocks, threads >> > (_Data, N * M);
	}

	template <class _Ty>
	void __softmax_apply(_Ty* _Data, size_t N, size_t M) {
		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((M + BLOCK_DIM - 1) / BLOCK_DIM);

		activate_softmax_kernel
			<<<blocks, threads>>>(_Data, N, M);
	}

	template <class _Ty>
	void _softmax_apply(_Ty* _Data, size_t N, size_t M) {

		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((N * M + BLOCK_DIM - 1) / BLOCK_DIM);

		using exponent = forwarder<cnn::exp>;

		for (size_t i = 0; i < M; ++i) {
			_Ty max = _range_max(_Data + i * N, N);
			cuda::_matrix_add_scalar(_Data + i * N, _Data + i * N, -max, N, size_t{ 1 });
		}

		activation_apply_kernel<exponent>
			<<<blocks, threads>>>(_Data, N * M);

		using applier = forwarder<cnn::softmax>;
		for (size_t i = 0; i < M; ++i) {
			const _Ty* sum = _range_reduce(_Data + i * N, N, 1);
			activation_apply_softmax_kernel<applier>
				<<<blocks, threads>>>(_Data + i * N, N, sum);
		}
	}

#define INSTANTIATE_ONE(_Fn, _Type) \
	template void _forward_apply<_Fn, _Type>(_Type*, size_t, size_t);\
	template void _backward_apply<_Fn, _Type>(_Type*, size_t, size_t);

#define INSTANTIATE_ACTIVATION_APPLY(_Fn) \
	INSTANTIATE_ONE(_Fn, float);\
	INSTANTIATE_ONE(_Fn, double);
	//INSTANTIATE_ONE(_Fn, bfloat16);

	INSTANTIATE_ACTIVATION_APPLY(cnn::relu);
	INSTANTIATE_ACTIVATION_APPLY(cnn::relu1);
	INSTANTIATE_ACTIVATION_APPLY(cnn::sigmoid);
	INSTANTIATE_ACTIVATION_APPLY(cnn::tanh);
	INSTANTIATE_ACTIVATION_APPLY(cnn::softmax);
	INSTANTIATE_ACTIVATION_APPLY(cnn::leaky_relu);
	INSTANTIATE_ACTIVATION_APPLY(cnn::log);
	INSTANTIATE_ACTIVATION_APPLY(cnn::sqrt);
	INSTANTIATE_ACTIVATION_APPLY(cnn::identity);
}