#pragma once
#include <immintrin.h>
#include <cassert>

namespace host {


    inline void avx2_mul_scalar(const float* _Src, float* _Dst, float _Val, size_t _Size) {
		__m256 _Vec_val = _mm256_set1_ps(_Val);
        for (size_t i = 0; i < _Size; i += 8) {
            __m256 _Vec_src = _mm256_load_ps(&_Src[i]);
            __m256 _Vec_res = _mm256_mul_ps(_Vec_src, _Vec_val);
            _mm256_store_ps(&_Dst[i], _Vec_res);
        }
    }

    inline void avx2_mul_scalar(const double* _Src, double* _Dst, double _Val, size_t _Size) {
		__m256d _Vec_val = _mm256_set1_pd(_Val);
		for (size_t i = 0; i < _Size; i += 4) {
			__m256d _Vec_src = _mm256_load_pd(&_Src[i]);
			__m256d _Vec_res = _mm256_mul_pd(_Vec_src, _Vec_val);
			_mm256_store_pd(&_Dst[i], _Vec_res);
		}
    }
    
    inline void avx2_add_scalar(const float* _Src, float* _Dst, float _Val, size_t _Size) {
        __m256 _Vec_val = _mm256_set1_ps(_Val);
        for (size_t i = 0; i < _Size; i += 8) {
            __m256 _Vec_src = _mm256_load_ps(&_Src[i]);
            __m256 _Vec_res = _mm256_add_ps(_Vec_src, _Vec_val);
            _mm256_store_ps(&_Dst[i], _Vec_res);
        }
    }

	inline void avx2_add_scalar(const double* _Src, double* _Dst, double _Val, size_t _Size) {
		__m256d _Vec_val = _mm256_set1_pd(_Val);
		for (size_t i = 0; i < _Size; i += 4) {
			__m256d _Vec_src = _mm256_load_pd(&_Src[i]);
			__m256d _Vec_res = _mm256_add_pd(_Vec_src, _Vec_val);
			_mm256_store_pd(&_Dst[i], _Vec_res);
		}
    }


					
}