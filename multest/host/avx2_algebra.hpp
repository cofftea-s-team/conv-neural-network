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
}