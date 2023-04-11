#pragma once
#include "utils_cuda.cuh"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

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
	inline void free(_Ty* ptr) {
		auto _Status = cudaFree((void*)ptr);

		if (_Status != cudaSuccess) {
			std::cout << "Cuda deallocation failed!" << std::endl;
			throw std::bad_alloc();
		}
	}
	
	template <class _Ty>
	inline void memcpy(const _Ty* _Src, _Ty* _Dst, size_t _Count, cudaOperationKind _Kind) {
		auto _Status = cudaMemcpy(_Dst, _Src, _Count * sizeof(_Ty), (cudaMemcpyKind)_Kind);

		if (_Status != cudaSuccess) {
			std::cout << "Copying memory failed!" << std::endl;
			throw std::exception("Copying memory failed!");
		}
	}

	template <class _Ty>
	inline void memcpy_transpose(const _Ty* _Src, _Ty* _Dst, size_t _Rows, size_t _Cols, cudaOperationKind _Kind) {
		size_t src_pitch = _Cols * sizeof(_Ty);
		size_t dst_pitch = _Rows * sizeof(_Ty);

		cudaMemcpy2D(_Dst, dst_pitch, _Src, src_pitch, _Cols * sizeof(_Ty), _Rows, (cudaMemcpyKind)_Kind);
	}

	template <class _Ty>
	inline _Ty* alloc_paged(size_t _Count) {
		_Ty* _Allocated_block;
		auto _Status = cudaMallocHost((void**)&_Allocated_block, _Count * sizeof(_Ty));

		if (_Status != cudaSuccess) {
			std::cout << "Cuda allocation failed!" << std::endl;
			throw std::bad_alloc();
		}

		return _Allocated_block;
	}
	
	template <class _Ty>
	inline void free_paged(_Ty* ptr) {
		auto _Status = cudaFreeHost((void*)ptr);

		if (_Status != cudaSuccess) {
			std::cout << "Cuda deallocation failed!" << std::endl;
			throw std::bad_alloc();
		}
	}
	
	template <class _Ty>
	inline _Ty from_cuda(const _Ty* _Val) {
		_Ty _Res;
		cuda::memcpy(_Val, &_Res, 1, DeviceToHost);
		return _Res;
	}

	template <class _Ty>
	inline void to_cuda(const _Ty* _Val, _Ty* _Dst) {
		cuda::memcpy(_Val, _Dst, 1, HostToDevice);
	}
};

