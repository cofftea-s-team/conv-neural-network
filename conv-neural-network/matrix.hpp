#pragma once
#include <iostream>
#include "cuda/algebra/matrix_transpose.cuh"
#include "host/algebra/matrix_transpose.hpp"
#include "cuda/utils.hpp"
#include "host/utils.hpp"

namespace base {
	struct shape
		: public std::pair<size_t, size_t>
	{
		using _Mybase = std::pair<size_t, size_t>;
		using _Mybase::_Mybase;
		inline size_t rows() const {
			return _Mybase::first;
		}
		inline size_t cols() const {
			return _Mybase::second;
		}

		inline constexpr shape transposed() const {
			return shape(cols(), rows());
		}

		inline constexpr shape T() const {
			return shape(cols(), rows());
		}

		inline friend std::ostream& operator<<(std::ostream& _Ostr, const shape& _Shape) {
			return _Ostr << "(" << _Shape.rows() << ", " << _Shape.cols() << ")";
		}
	};
	

	template <class _Ty, bool _Is_cuda>
	struct allocator {
		using value_type = _Ty;
		virtual _Ty* alloc(size_t _Count) const = 0;
		virtual void free(_Ty* ptr) const = 0;
		static constexpr bool is_cuda() {
			return _Is_cuda;
		}
	};

	template <typename T>
	concept allocator_t = std::is_base_of_v<allocator<typename T::value_type, T::is_cuda()>, T>;

	template <class _Ty, allocator_t, bool>
	class vector;
	
	template <class _Ty, allocator_t _Alloc, bool _T>
	class matrix {
	public:
		using value_type = _Ty;
		using alloc = _Alloc;
		using iterator = std::conditional_t<_Alloc::is_cuda(), cuda::cuda_to_host_matrix_iterator<_Ty>, _Ty*>;
		using const_iterator = std::conditional_t<_Alloc::is_cuda(), cuda::cuda_to_host_matrix_iterator<const _Ty>, const _Ty*>;

		template <class _Ty2, allocator_t _All2, bool _T2>
		friend class matrix;


		inline matrix(size_t _M, size_t _N) 
			: _Rows(_M), _Cols(_N) 
		{
			_Data = _Al.alloc(_N * _M);
		}

		inline matrix(const shape& _Shape)
			: matrix(_Shape.rows(), _Shape.cols())
		{ }
		
		template <allocator_t _Other_all, bool _T2>
		inline matrix(const matrix<_Ty, _Other_all, _T2>& _Other) {
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Data = _Al.alloc(_Rows * _Cols);
			_Copy_matrix(_Other);
		}

		inline matrix(const matrix& _Other) {
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Data = _Al.alloc(_Rows * _Cols);
			_Copy_matrix(_Other);
		}

		inline matrix(matrix&& _Other) noexcept {
			_Move(std::move(_Other));
		}

		inline matrix(_Ty* _Data, size_t _Rows, size_t _Cols)
			: _Rows(_Cols), _Cols(_Rows), _Data(_Data), _Is_owner(false)
		{ }

		template <bool _T2>
		inline matrix(const matrix<_Ty, _Alloc, _T2>& _Other) {
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Data = _Al.alloc(_Rows * _Cols);
			_Copy_matrix(_Other);
		}
		
		inline ~matrix() {
			if (_Is_owner) {
				_Al.free(_Data);
				_Data = nullptr;
			}
		}

		virtual _Ty& operator()(size_t _I, size_t _J) {
			return _Data[_I * _Cols + _J];
		}

		virtual const _Ty& operator()(size_t _I, size_t _J) const {
			return _Data[_I * _Cols + _J];
		}

		template <allocator_t _Other_all, bool _T2>
		inline matrix& operator=(const matrix<_Ty, _Other_all, _T2>& _Other) {
			if (this == &_Other) {
				return *this;
			}
			if (_Rows * _Cols != _Other._Rows * _Other._Cols) {
				_Al.free(_Data);
				_Data = _Al.alloc(_Other._Rows * _Other._Cols);
			}
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Copy_matrix(_Other);
			
			return *this;
		}

		template <allocator_t _Other_all>
		inline matrix& operator=(matrix<_Ty, _Other_all, _T>&& _Other) {
			if (this == &_Other) {
				return *this;
			}
			if (_Is_owner) {
				_Al.free(_Data);
			}
			_Move(std::move(_Other));
			return *this;
		}

		inline matrix& operator=(const matrix& _Other) {
			if (_Rows * _Cols != _Other._Rows * _Other._Cols) {
				_Al.free(_Data);
				_Data = _Al.alloc(_Other._Rows * _Other._Cols);
			}
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Copy_matrix(_Other);

			return *this;
		}

		inline matrix& operator=(matrix&& _Other) noexcept {
			if (this == &_Other) {
				return *this;
			}
			if (_Is_owner && _Data != nullptr) {
				_Al.free(_Data);
			}
			_Move(std::move(_Other));

			return *this;
		}
		
		inline _Ty* data() {
			return _Data;
		}

		inline const _Ty* data() const {
			return _Data;
		}

		inline size_t rows() const {
			return _Rows;
		}

		inline size_t cols() const {
 			return _Cols;
		}

		inline shape shape() const {
			return { _Rows, _Cols };
		}

		inline size_t size() const {
			return _Rows * _Cols;
		}

		inline iterator begin() {
			return iterator(_Data);
		}

		inline const_iterator begin() const {
			return const_iterator(_Data);
		}

		inline iterator end() {
			return iterator(_Data + _Rows * _Cols);
		}

		inline const_iterator end() const {
			return const_iterator(_Data + _Rows * _Cols);
		}
		
		constexpr static _Alloc get_allocator() {
			return _Al;
		}

		constexpr static bool is_transposed() {
			return _T;
		}
		
	protected:
		inline matrix() = default;

		_Ty* _Data = nullptr;
		size_t _Rows;
		size_t _Cols;
		bool _Is_owner = true;
		constexpr static _Alloc _Al = {};
		
	private:
		template <allocator_t _Other_all, bool _T2>
		inline void _Copy_matrix(const matrix<_Ty, _Other_all, _T2>& _M) {
			if constexpr (_M.is_transposed()) {
				if constexpr (_Al.is_cuda())
					_Copy_to_cuda_transposed(_M);
				else
					_Copy_to_host_transposed(_M);
				return;
			}
			if constexpr (_Al.is_cuda())
				_Copy_to_cuda(_M);
			else
				_Copy_to_host(_M);
		}
		
		template <allocator_t _Other_all, bool _T2>
		constexpr void _Copy_to_cuda(const matrix<_Ty, _Other_all, _T2>& _Other) {
			if constexpr (_Other._Al.is_cuda()) 
				cuda::memcpy(_Other._Data, _Data, _Rows * _Cols, DeviceToDevice);
			else
				cuda::memcpy(_Other._Data, _Data, _Rows * _Cols, HostToDevice);
		}

		template <allocator_t _Other_all, bool _T2>
		constexpr void _Copy_to_host(const matrix<_Ty, _Other_all, _T2>& _Other) {
			if constexpr (_Other._Al.is_cuda())
				cuda::memcpy(_Other._Data, _Data, _Rows * _Cols, DeviceToHost);
			else
				cuda::memcpy(_Other._Data, _Data, _Rows * _Cols, HostToHost);
		}

		template <allocator_t _Other_all, bool _T2>
		constexpr void _Copy_to_cuda_transposed(const matrix<_Ty, _Other_all, _T2>& _Other) {
			if constexpr (_Other._Al.is_cuda())
				cuda::matrix_copy_transposed(_Other, *this);
			else
				static_assert(std::_Always_false<matrix>, "not implemented!");
		}
		
		template <allocator_t _Other_all, bool _T2>	
		constexpr void _Copy_to_host_transposed(const matrix<_Ty, _Other_all, _T2>& _Other) {
			if constexpr (_Other._Al.is_cuda())
				static_assert(std::_Always_false<matrix>, "not implemented!");
			else
				host::matrix_transpose(_Other, *this);
		}

		template<allocator_t _Other_all, bool _T2>
		constexpr void _Move(base::matrix<base::matrix<_Ty, _Alloc, _T>::value_type, _Other_all, _T2>&& _Other) noexcept {
			if (this == &_Other) {
				return;
			}
			_Rows = _Other._Rows;
			_Cols = _Other._Cols;
			_Data = _Other._Data;
			_Is_owner = _Other._Is_owner;
			_Other._Is_owner = false;
			_Other._Data = nullptr;
		}
	};
}