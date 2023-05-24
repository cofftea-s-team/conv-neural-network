#pragma once
#include "../algebra/utils.hpp"
#include <algorithm>

namespace host::activations {
	
	namespace parallel {
		
		template <class _Ty>
		inline void softmax_forward(_Ty* _Data, size_t N, size_t M) {
			algebra::parallel::parallel_for(M, [=](size_t i) {
				auto _Max = *std::max_element(std::next(_Data, i * N), std::next(_Data, i * N + N));
				_Ty _Row_sum = 0.0;
				for (size_t j = 0; j < N; ++j) {
					_Data[i * N + j] = std::exp(_Data[i * N + j] - _Max);
					_Row_sum += _Data[i * N + j];
				}

				for (size_t j = 0; j < N; ++j) {
					_Data[i * N + j] /= _Row_sum;
				}
			});
		}
	}

	template <class _Ty>
	inline void softmax_forward(_Ty* _Data, size_t N, size_t M) {
		for (size_t i = 0; i < M; ++i) {
			auto _Max = *std::max_element(std::next(_Data, i * N), std::next(_Data, i * N + N));
			_Ty _Row_sum = 0.0;
			for (size_t j = 0; j < N; ++j) {
				_Data[i * N + j] = std::exp(_Data[i * N + j] - _Max);
				_Row_sum += _Data[i * N + j];
			}
			
			for (size_t j = 0; j < N; ++j) {
				_Data[i * N + j] /= _Row_sum;
			}
		}
	}
}

namespace host {
	
	template <class _Mat>
	inline void activation_apply_softmax(_Mat& _M) {
		auto _Data = _M.data();
		size_t N = _M.cols();
		size_t M = _M.rows();

		if (N < 128 || M < 16) {
			host::activations::softmax_forward(_Data, N, M);
		}
		else {
			host::activations::parallel::softmax_forward(_Data, N, M);
		}
	}
}