#include "activation.cuh"

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

			const dim3 threads(512);
			const dim3 blocks((N + threads.x - 1) / threads.x);

			activation_apply_kernel<applier>
				<<<blocks, threads>>>(_Data, N * M);
		}
	}

	template <class _Activation_class, class _Ty>
	void _backward_apply(_Ty* _Data, size_t N, size_t M) {
		using applier = backwarder<_Activation_class>;

		const dim3 threads(512);
		const dim3 blocks((N + threads.x - 1) / threads.x);

		activation_apply_kernel<applier>
			<< <blocks, threads >> > (_Data, N * M);
	}

	struct _exp {
		template <class _Ty>
		__device__ inline static _Ty forward(_Ty x) {
			return exp(x);
		}
	};

	template <class _Ty>
	void _softmax_apply(_Ty* _Data, size_t N, size_t M) {

		const dim3 threads(256);
		const dim3 blocks((N + threads.x - 1) / threads.x);

		using exponent = forwarder<_exp>;

		activation_apply_kernel<exponent>
			<<<blocks, threads>>>(_Data, N * M);

		_Ty sum = _range_reduce(_Data, N, M);

		using applier = forwarder<cnn::softmax>;
		activation_apply_kernel<applier>
			<<<blocks, threads>>>(_Data, N * M, sum);
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
}