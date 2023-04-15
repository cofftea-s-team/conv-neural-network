#pragma once
#include <immintrin.h>
#include <cassert>

namespace host {
    inline void transpose4x4_SSE(const float* _Src, float* _Dst, const size_t N, const size_t M) {
        __m128 row1 = _mm_load_ps(&_Src[0 * N]);
        __m128 row2 = _mm_load_ps(&_Src[1 * N]);
        __m128 row3 = _mm_load_ps(&_Src[2 * N]);
        __m128 row4 = _mm_load_ps(&_Src[3 * N]);
        _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
        _mm_store_ps(&_Dst[0 * M], row1);
        _mm_store_ps(&_Dst[1 * M], row2);
        _mm_store_ps(&_Dst[2 * M], row3);
        _mm_store_ps(&_Dst[3 * M], row4);
    }

    template <size_t _Block_size = 16>
    inline void avx2_transpose(const float* _Src, float* _Dst, size_t _Rows, size_t _Cols) {
        for (size_t i = 0; i < _Rows ; i += _Block_size) {
            for (size_t j = 0; j < _Cols; j += _Block_size) {
                size_t max_i2 = i + _Block_size < _Rows ? i + _Block_size : _Rows;
                size_t max_j2 = j + _Block_size < _Cols ? j + _Block_size : _Cols;
                for (size_t i2 = i; i2 < max_i2; i2 += 4) {
                    for (size_t j2 = j; j2 < max_j2; j2 += 4) {
                        transpose4x4_SSE(&_Src[i2 * _Cols + j2], &_Dst[j2 * _Rows + i2], _Cols, _Rows);
                    }
                }
            }
        }
    }

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

    template <bool _T1, bool _T2>
	inline void avx2_add_vector(const float* _Src, const float* _Vec, float* _Dst, size_t N, size_t M) { // M - row count, N - column count
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; j += 8) {
                
                __m256 _Vec_src = _mm256_set_ps(
                    _Src[i + (j+0) * N],
                    _Src[i + (j+1) * N],
                    _Src[i + (j+2) * N],
                    _Src[i + (j+3) * N], 
                    _Src[i + (j+4) * N], 
                    _Src[i + (j+5) * N], 
                    _Src[i + (j+6) * N], 
                    _Src[i + (j+7) * N]  
                );

				__m256 _Vec_vec = _mm256_load_ps(&_Vec[j]);
                float _Tmp[8];
				__m256 _Vec_res = _mm256_add_ps(_Vec_src, _Vec_vec);
				_mm256_store_ps(_Tmp, _Vec_src);
				for (size_t k = 0; k < 8; ++k) {
					_Dst[i + (j+k)*N] = _Tmp[k] + _Vec[j + k];
				}
            }
        }
    }


					
}