#include "activation.cuh"

namespace cuda {
	
	template <class _Activation_class, class _Ty>
	__global__ void activation_apply_kernel(_Ty* _Data, size_t N) {
		const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < N) {
			_Data[i] = _Activation_class::forward(_Data[i]);
		}
	}

	template <class _Activation_class, class _Ty>
	void _activation_apply(_Ty* _Data, size_t N, size_t M) {
		if constexpr (std::is_same_v<_Activation_class, base::softmax>) {
			_softmax_apply(_Data, N, M);
		}
		else {
			const dim3 threads(256);
			const dim3 blocks((N + threads.x - 1) / threads.x);

			activation_apply_kernel<_Activation_class>
				<<<blocks, threads>>>(_Data, N * M);
		}

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

		activation_apply_kernel<_exp>
			<<<blocks, threads>>>(_Data, N * M);

		_Ty sum = _range_reduce(_Data, N, M);

		activation_apply_kernel<base::softmax>
			<<<blocks, threads>>>(_Data, N * M);
	}

	template void _activation_apply<base::relu, double>(double*, size_t);
	template void _activation_apply<base::relu, float>(float*, size_t);
	template void _activation_apply<base::relu, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<base::sigmoid, double>(double*, size_t);
	template void _activation_apply<base::sigmoid, float>(float*, size_t);
	template void _activation_apply<base::sigmoid, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<base::tanh, double>(double*, size_t);
	template void _activation_apply<base::tanh, float>(float*, size_t);
	template void _activation_apply<base::tanh, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<base::softmax, double>(double*, size_t);
	template void _activation_apply<base::softmax, float>(float*, size_t);
	template void _activation_apply<base::softmax, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<base::leaky_relu, double>(double*, size_t);
	template void _activation_apply<base::leaky_relu, float>(float*, size_t);
	template void _activation_apply<base::leaky_relu, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<base::softmax, double>(double*, size_t);
	template void _activation_apply<base::softmax, float>(float*, size_t);
	template void _activation_apply<base::softmax, nv_bfloat16>(nv_bfloat16*, size_t);
	
}