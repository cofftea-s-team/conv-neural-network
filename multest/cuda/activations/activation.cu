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
	void _activation_apply(_Ty* _Data, size_t N) {
		
		const dim3 threads(256);
		const dim3 blocks((N + threads.x - 1) / threads.x);

		activation_apply_kernel<_Activation_class>
			<<<blocks, threads>>>(_Data, N);

	}

	using namespace base;
	template void _activation_apply<relu, double>(double*, size_t);
	template void _activation_apply<relu, float>(float*, size_t);
	template void _activation_apply<relu, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<sigmoid, double>(double*, size_t);
	template void _activation_apply<sigmoid, float>(float*, size_t);
	template void _activation_apply<sigmoid, nv_bfloat16>(nv_bfloat16*, size_t);

	template void _activation_apply<base::tanh, double>(double*, size_t);
	template void _activation_apply<base::tanh, float>(float*, size_t);
	template void _activation_apply<base::tanh, nv_bfloat16>(nv_bfloat16*, size_t);
}