#include "activation.cuh"

#define BLOCK_DIM 1024

namespace cuda {
	
	template <class _Applier, class _Ty, class... _TArgs>
	__global__ void activation_apply_kernel(_Ty* _Data, size_t N, _TArgs... _Args) {
		const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < N) {
			_Data[i] = _Applier::apply(_Data[i], _Args...);
		}
	}

	template <class _Activation_class, class _Ty>
	void _forward_apply(_Ty* _Data, size_t N, size_t M) {
		if constexpr (std::is_same_v<_Activation_class, cnn::softmax>) {
			_softmax_apply(_Data, N, M);
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
	void _softmax_apply(_Ty* _Data, size_t N, size_t M) {

		const dim3 threads(BLOCK_DIM);
		const dim3 blocks((N * M + BLOCK_DIM - 1) / BLOCK_DIM);

		using exponent = forwarder<cnn::exp>;

		activation_apply_kernel<exponent>
			<<<blocks, threads>>>(_Data, N * M);

		using applier = forwarder<cnn::softmax>;
		for (size_t i = 0; i < M; ++i) {
			_Ty sum = _range_reduce(_Data + i * N, N, 1);

			activation_apply_kernel<applier>
				<<<blocks, threads>>>(_Data + i * N, N, sum);
		}
	}

#define INSTANTIATE_ONE(_Fn, _Type) \
	template void _forward_apply<_Fn, _Type>(_Type*, size_t, size_t);\
	template void _backward_apply<_Fn, _Type>(_Type*, size_t, size_t);

#define INSTANTIATE_ACTIVATION_APPLY(_Fn) \
	INSTANTIATE_ONE(_Fn, float);\
	INSTANTIATE_ONE(_Fn, double);\
	INSTANTIATE_ONE(_Fn, bfloat16);

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