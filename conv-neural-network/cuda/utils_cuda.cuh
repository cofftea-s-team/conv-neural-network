#pragma once
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda_bf16.h"
#include "curand.h"
#include "curand_kernel.h"
#include <iostream>
#include "memory.hpp"

using bfloat16 = nv_bfloat16;
enum RAND_MODE {
	UNIFORM,
	NORMAL
};

enum cudaOperationKind {
	HostToDevice = cudaMemcpyHostToDevice,
	HostToHost = cudaMemcpyHostToHost,
	DeviceToHost = cudaMemcpyDeviceToHost,
	DeviceToDevice = cudaMemcpyDeviceToDevice
};

namespace cuda {

	template <class _Ty>
	inline _Ty* alloc(size_t _Count) {
		_Ty* _Allocated_block;
		auto _Status = cudaMalloc((void**)&_Allocated_block, _Count * sizeof(_Ty));

		if (_Status != cudaSuccess) {
			std::cout << "Cuda allocation failed!" << std::endl;
			throw std::bad_alloc();
		}

		return _Allocated_block;
	}

	template <class _Ty>
	inline void free(_Ty* _Ptr) {
		if (_Ptr == nullptr) {
			return;
		}

		auto _Status = cudaFree((void*)_Ptr);

		if (_Status != cudaSuccess) {
			std::cout << "Cuda deallocation failed! " << _Ptr << std::endl;
			throw std::bad_alloc();
		}
	}

	template <class _Ty>
	inline void memcpy(const _Ty* _Src, _Ty* _Dst, size_t _Count, cudaOperationKind _Kind) {
		auto _Status = cudaMemcpy((void*)_Dst, (const void*)_Src, _Count * sizeof(_Ty), (cudaMemcpyKind)_Kind);

		if (_Status != cudaSuccess) {
			std::cout << "Copying memory failed!" << std::endl;
			throw std::exception("Copying memory failed!");
		}
	}

	template <RAND_MODE _Mode, class _Ty>
	void _fill_random(_Ty* _Data, size_t N);

	template <RAND_MODE _Mode, class _Mat>
	inline void fill_random(_Mat& _M) {
		_fill_random<_Mode>(_M.data(), _M.size());
	}
}