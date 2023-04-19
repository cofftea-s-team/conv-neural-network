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

namespace base {
	template <class _Ty, bool>
	struct allocator;
}

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
		auto _Status = cudaFree((void*)_Ptr);

		if (_Status != cudaSuccess) {
			std::cout << "Cuda deallocation failed!" << std::endl;
			throw std::bad_alloc();
		}
	}
	
	template <class _Ty>
	struct allocator
		: public base::allocator<_Ty, true>
	{
		constexpr _Ty* alloc(size_t _Count) const override {
			return cuda::alloc<_Ty>(_Count);
		}

		constexpr void free(_Ty* _Ptr) const override {
			cuda::free<_Ty>(_Ptr);
		}
	};

	template <class _Ty>
	inline void memcpy(const _Ty* _Src, _Ty* _Dst, size_t _Count, cudaOperationKind _Kind) {
		auto _Status = cudaMemcpy((void*)_Dst, (const void*)_Src, _Count * sizeof(_Ty), (cudaMemcpyKind)_Kind);

		if (_Status != cudaSuccess) {
			std::cout << "Copying memory failed!" << std::endl;
			throw std::exception("Copying memory failed!");
		}
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
	inline void free_paged(_Ty* _Ptr) {
		auto _Status = cudaFreeHost((void*)_Ptr);

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

	template <class _Ty>
	struct cuda_to_host_matrix_iterator {
		struct element {
			inline element(_Ty* _Ptr)
				: _Ptr(_Ptr)
			{ }

			inline element& operator=(const _Ty& _Val) {
				cuda::to_cuda(&_Val, _Ptr);
				return *this;
			}

			inline operator _Ty() const {
				return cuda::from_cuda(_Ptr);
			}

			inline friend std::ostream& operator<<(std::ostream& _Os, const element& _El) {
				_Os << cuda::from_cuda(_El._Ptr);
				return _Os;
			}

			_Ty* _Ptr;
		};
		inline cuda_to_host_matrix_iterator(_Ty* _Ptr)
			: _Ptr(_Ptr)
		{}

		inline cuda_to_host_matrix_iterator& operator++() {
			++_Ptr;
			return *this;
		}

		inline cuda_to_host_matrix_iterator operator++(int) {
			cuda_to_host_matrix_iterator _Tmp = *this;
			++_Ptr;
			return _Tmp;
		}

		inline cuda_to_host_matrix_iterator& operator--() {
			--_Ptr;
			return *this;
		}

		inline cuda_to_host_matrix_iterator operator--(int) {
			cuda_to_host_matrix_iterator _Tmp = *this;
			--_Ptr;
			return _Tmp;
		}

		inline cuda_to_host_matrix_iterator& operator+=(int _Off) {
			_Ptr += _Off;
			return *this;
		}

		inline cuda_to_host_matrix_iterator& operator-=(int _Off) {
			_Ptr -= _Off;
			return *this;
		}

		inline cuda_to_host_matrix_iterator operator+(int _Off) const {
			return cuda_to_host_matrix_iterator(_Ptr + _Off);
		}

		inline cuda_to_host_matrix_iterator operator-(int _Off) const {
			return cuda_to_host_matrix_iterator(_Ptr - _Off);
		}

		inline int operator-(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr - _Other._Ptr;
		}

		inline bool operator==(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr == _Other._Ptr;
		}

		inline bool operator!=(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr != _Other._Ptr;
		}

		inline bool operator<(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr < _Other._Ptr;
		}

		inline bool operator>(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr > _Other._Ptr;
		}

		inline bool operator<=(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr <= _Other._Ptr;
		}

		inline bool operator>=(const cuda_to_host_matrix_iterator& _Other) const {
			return _Ptr >= _Other._Ptr;
		}

		inline element operator*() const {
			return element(_Ptr);
		}
	private:
		_Ty* _Ptr;
	};
};

