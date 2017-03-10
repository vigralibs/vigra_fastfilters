/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2017 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA2 computer vision library.          */
/*    The VIGRA2 Website is                                             */
/*        http://ukoethe.github.io/vigra2                               */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#pragma once

#ifndef VIGRA2_FASTFILTERS_HXX
#define VIGRA2_FASTFILTERS_HXX

#include <vigra2/config.hxx>
#include <vigra2/numeric_traits.hxx>
#include <vigra2/array_nd.hxx>
#include <immintrin.h>
#include <cassert>
#include <stdexcept>

// FIXME: may not be needed (no speedup observed)
//#ifdef HAVE_BUILTIN_EXPECT
#ifndef _MSC_VER
#define likely(x) __builtin_expect(x, true)
#define unlikely(x) __builtin_expect(x, false)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#ifndef __FMA__
#define _mm256_fmadd_ps(a, b, c) (_mm256_add_ps(_mm256_mul_ps((a), (b)), (c)))
#endif


namespace vigra {

/** \addtogroup Filters
*/
//@{


#define ALIGN_MAGIC 0xd2ac461d9c25ee00

void *fastfilters_memory_align(size_t alignment, size_t size)
{
    assert(alignment < 0xff);
    void *ptr = malloc(size + alignment + 8);

    if (!ptr)
        return NULL;

    uintptr_t ptr_i = (uintptr_t)ptr;
    ptr_i += 8 + alignment - 1;
    ptr_i &= ~(alignment - 1);

    uintptr_t ptr_diff = ptr_i - (uintptr_t)ptr;

    void *ptr_aligned = (void *)ptr_i;
    uint64_t *ptr_magic = (uint64_t *)(ptr_i - 8);

    *ptr_magic = ALIGN_MAGIC | (ptr_diff & 0xff);

    return ptr_aligned;
}

void fastfilters_memory_align_free(void *vptr)
{
    char * ptr = (char*)vptr;
    uint64_t magic = *(uint64_t *)(ptr - 8);

    assert((magic & ~0xff) == ALIGN_MAGIC);

    ptr -= magic & 0xff;
    free(ptr);
}

enum KernelSymmetry { KernelEven, KernelOdd, KernelNotSymmetric };

template <KernelSymmetry SYMMETRY = KernelEven>
struct AddBySymmetry
{
    template <class T>
    T operator()(T a, T b) const
    {
        return a + b;
    }

    __m256 operator()(__m256 a, __m256 b) const
    {
        return _mm256_add_ps(a, b);
    }
};

template <>
struct AddBySymmetry<KernelOdd>
{
    template <class T>
    T operator()(T a, T b) const
    {
        return a - b;
    }

    __m256 operator()(__m256 a, __m256 b) const
    {
        return _mm256_sub_ps(a, b);
    }
};

template <ArrayIndex RADIUS>
struct RadiusLoopUnrolling
{
    constexpr ArrayIndex operator()(ArrayIndex) const
    {
        return RADIUS;
    }
};

template <>
struct RadiusLoopUnrolling<runtime_size>
{
    ArrayIndex operator()(ArrayIndex radius) const
    {
        return radius;
    }
};

template <KernelSymmetry SYMMETRY = KernelEven, ArrayIndex RADIUS = runtime_size>
struct AvxConvolveLine
{
    static void exec_x(float * in, ArrayIndex size, float * out,
                       float * kernel, ArrayIndex radius)
    {
        AddBySymmetry<SYMMETRY> addsub;
        RadiusLoopUnrolling<RADIUS> const_radius;

        // FIXME: check kernel longer than line
        ArrayIndex x = 0;
        for (; x < const_radius(radius); ++x)
        {
            float sum = kernel[0] * in[x];

            for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
            {
                ArrayIndex offset_left  = x < k ? k - x
                                                : x - k,
                           offset_right = likely(k + x < size) ? x + k
                                                               : size - ((k + x) % size) - 2;

                sum += kernel[k] * addsub(in[offset_right], in[offset_left]);
            }

            out[x] = sum;
        }

        // FIXME: check if this must be radius + 1
        ArrayIndex steps = (size - const_radius(radius) - x) / 32;
        for(ArrayIndex j = 0; j < steps; ++j, x += 32)
        {
            // load next 32 pixels
            __m256 result0 = _mm256_loadu_ps(in + x);
            __m256 result1 = _mm256_loadu_ps(in + x + 8);
            __m256 result2 = _mm256_loadu_ps(in + x + 16);
            __m256 result3 = _mm256_loadu_ps(in + x + 24);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_broadcast_ss(kernel);
            result0 = _mm256_mul_ps(result0, kernel_val);
            result1 = _mm256_mul_ps(result1, kernel_val);
            result2 = _mm256_mul_ps(result2, kernel_val);
            result3 = _mm256_mul_ps(result3, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            // FIXME: consider explicit loop unrolling
            for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
            {
                kernel_val = _mm256_broadcast_ss(kernel + k);

                // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] =
                // (image[i-j] +
                // image[i+j]) * kernel[j])
                // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
                __m256 pixels0 = addsub(_mm256_loadu_ps(in + x + k     ), _mm256_loadu_ps(in + (x - k)));
                __m256 pixels1 = addsub(_mm256_loadu_ps(in + x + k +  8), _mm256_loadu_ps(in + (x - k) + 8));
                __m256 pixels2 = addsub(_mm256_loadu_ps(in + x + k + 16), _mm256_loadu_ps(in + (x - k) + 16));
                __m256 pixels3 = addsub(_mm256_loadu_ps(in + x + k + 24), _mm256_loadu_ps(in + (x - k) + 24));

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
                result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
                result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
                result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
            }

            _mm256_storeu_ps(out + x, result0);
            _mm256_storeu_ps(out + x + 8, result1);
            _mm256_storeu_ps(out + x + 16, result2);
            _mm256_storeu_ps(out + x + 24, result3);
        }

        // FIXME: check if this must be radius + 1
        steps = (size - const_radius(radius) - x) / 8;
        for (ArrayIndex j = 0; j < steps; ++j, x += 8)
        {
            // load next 8 pixels
            __m256 result0 = _mm256_loadu_ps(in + x);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_broadcast_ss(kernel);
            result0 = _mm256_mul_ps(result0, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
            {
                kernel_val = _mm256_broadcast_ss(kernel + k);

                __m256 pixels0 = addsub(_mm256_loadu_ps(in + x + k), _mm256_loadu_ps(in + (x - k)));

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
            }
            _mm256_storeu_ps(out + x, result0);
        }

        // FIXME: check if this must be radius + 1
        for (; x < size - const_radius(radius); ++x)
        {
            float sum = kernel[0] * in[x];

            // work on both sides of symmetric kernel simultaneously
            for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
            {
                sum += kernel[k] * addsub(in[x + k], in[x - k]);
            }
            out[x] = sum;
        }

        // right border
        for (; x < size; ++x)
        {
            float sum = in[x] * kernel[0];

            for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
            {
                float left = unlikely(x < k) ? in[k - x]
                                             : in[x - k],
                      right = x + k >= size ? in[size - ((k + x) % size) - 2]
                                            : in[x + k];

                sum += kernel[k] * addsub(right, left);
            }

            out[x] = sum;
        }
    }

    static void exec_y(float * in, ArrayIndex width, ArrayIndex height, ArrayIndex in_stride,
                       float * out, ArrayIndex out_stride,
                       float * kernel, ArrayIndex radius)
    {
        AddBySymmetry<SYMMETRY> addsub;
        RadiusLoopUnrolling<RADIUS> const_radius;

        // FIXME: check kernel longer than line

        const size_t avx_end = (size_t)width & ~7; // round down to nearest multiple of 8
        const size_t noavx_left = (size_t)width - avx_end;
        const size_t n_outer_aligned = avx_end + 8;

        const __m256i mask =
            _mm256_set_epi32(0, noavx_left >= 7 ? 0xffffffff : 0, noavx_left >= 6 ? 0xffffffff : 0,
                             noavx_left >= 5 ? 0xffffffff : 0, noavx_left >= 4 ? 0xffffffff : 0,
                             noavx_left >= 3 ? 0xffffffff : 0, noavx_left >= 2 ? 0xffffffff : 0, 0xffffffff);

        float * tmp = (float*)fastfilters_memory_align(32, (radius + 1) * n_outer_aligned * sizeof(float));
        if(!tmp)
            throw std::bad_alloc();

        size_t y = 0;

        // left border
        for (; y < const_radius(radius); ++y)
        {
            const float *cur_inptr = in + y * in_stride;
            const size_t tmpidx = y % (const_radius(radius) + 1);
            float *tmpptr = tmp + tmpidx * n_outer_aligned;

            size_t x;
            for (x = 0; x < avx_end; x += 8)
            {
                __m256 pixels = _mm256_loadu_ps(cur_inptr + x);
                __m256 kernel_val = _mm256_broadcast_ss(kernel);
                __m256 result = _mm256_mul_ps(pixels, kernel_val);

                for (unsigned int k = 1; k <= const_radius(radius); ++k)
                {
                    kernel_val = _mm256_broadcast_ss(kernel + k);
                    __m256 pixel_left = (k > y) ? _mm256_loadu_ps(in + (k - y) * in_stride + x)
                                                : _mm256_loadu_ps(in + (y - k) * in_stride + x);

                    __m256 pixels_right =
                         likely(y + k < height) ? _mm256_loadu_ps(in + (y + k) * in_stride + x)
                                                : _mm256_loadu_ps(in + (height - ((k + y) % height) - 2) * in_stride + x);

                    pixels = addsub(pixels_right, pixel_left);
                    result = _mm256_fmadd_ps(pixels, kernel_val, result);
                }

                _mm256_store_ps(tmpptr + x, result);
            }

            if (noavx_left > 0)
            {
                __m256 pixels = _mm256_maskload_ps(cur_inptr + x, mask);
                __m256 kernel_val = _mm256_broadcast_ss(kernel);
                __m256 result = _mm256_mul_ps(pixels, kernel_val);

                for (unsigned int k = 1; k <= const_radius(radius); ++k)
                {
                    kernel_val = _mm256_broadcast_ss(kernel + k);
                    __m256 pixel_left = (k > y)
                          ? _mm256_maskload_ps(in + (k - y) * in_stride + x, mask)
                          : _mm256_maskload_ps(in + (y - k) * in_stride + x, mask);

                    __m256 pixels_right = likely(y + k < height)
                        ? _mm256_maskload_ps(in + (y + k) * in_stride + x, mask)
                        : _mm256_maskload_ps(in + (height - ((k + y) % height) - 2) * in_stride + x, mask);

                    pixels = addsub(pixels_right, pixel_left);
                    result = _mm256_fmadd_ps(pixels, kernel_val, result);
                }

                _mm256_store_ps(tmpptr + x, result);
            }
        }

        // valid part of line
        const size_t y_end = height - const_radius(radius);

        for (; y < y_end; ++y)
        {
            const float *cur_inptr = in + y * in_stride;
            const unsigned tmpidx = y % (const_radius(radius) + 1);
            float *tmpptr = tmp + tmpidx * n_outer_aligned;

            unsigned x;
            for (x = 0; x < avx_end; x += 8)
            {
                __m256 pixels = _mm256_loadu_ps(cur_inptr + x);
                __m256 kernel_val = _mm256_broadcast_ss(kernel);
                __m256 result = _mm256_mul_ps(pixels, kernel_val);

                for (unsigned int k = 1; k <= const_radius(radius); ++k)
                {
                    kernel_val = _mm256_broadcast_ss(kernel + k);

                    pixels = addsub(_mm256_loadu_ps(in + (y + k) * in_stride + x),
                                    _mm256_loadu_ps(in + (y - k) * in_stride + x));
                    result = _mm256_fmadd_ps(pixels, kernel_val, result);
                }

                _mm256_store_ps(tmpptr + x, result);
            }

            if (noavx_left > 0)
            {
                __m256 pixels = _mm256_maskload_ps(cur_inptr + x, mask);
                __m256 kernel_val = _mm256_broadcast_ss(kernel);
                __m256 result = _mm256_mul_ps(pixels, kernel_val);

                for (unsigned int k = 1; k <= const_radius(radius); ++k)
                {
                    kernel_val = _mm256_broadcast_ss(kernel + k);

                    pixels = addsub(_mm256_maskload_ps(in + (y + k) * in_stride + x, mask),
                                    _mm256_maskload_ps(in + (y - k) * in_stride + x, mask));
                    result = _mm256_fmadd_ps(pixels, kernel_val, result);
                }

                _mm256_store_ps(tmpptr + x, result);
            }

            const unsigned writeidx = (y + 1) % (const_radius(radius) + 1);
            float *writeptr = tmp + writeidx * n_outer_aligned;
            memcpy(out + (y - const_radius(radius)) * out_stride, writeptr, width * sizeof(float));
        }

        // right border
        for (; y < height; ++y)
        {
            const float *cur_inptr = in + y * in_stride;
            const unsigned tmpidx = y % (const_radius(radius) + 1);
            float *tmpptr = tmp + tmpidx * n_outer_aligned;

            unsigned x;
            for (x = 0; x < avx_end; x += 8)
            {
                __m256 pixels = _mm256_loadu_ps(cur_inptr + x);
                __m256 kernel_val = _mm256_broadcast_ss(kernel);
                __m256 result = _mm256_mul_ps(pixels, kernel_val);

                for (unsigned int k = 1; k <= const_radius(radius); ++k)
                {
                    kernel_val = _mm256_broadcast_ss(kernel + k);
                    __m256 pixel_right = (y + k < height)
                        ? _mm256_loadu_ps(in + (y + k) * in_stride + x)
                        : _mm256_loadu_ps(in + (height - ((k + y) % height) - 2) * in_stride + x);

                    __m256 pixel_left = (k > y)
                         ? _mm256_loadu_ps(in + (k - y) * in_stride + x)
                         : _mm256_loadu_ps(in + (y - k) * in_stride + x);

                    pixels = addsub(pixel_right, pixel_left);
                    result = _mm256_fmadd_ps(pixels, kernel_val, result);
                }

                _mm256_store_ps(tmpptr + x, result);
            }

            if (noavx_left > 0)
            {
                __m256 pixels = _mm256_maskload_ps(cur_inptr + x, mask);
                __m256 kernel_val = _mm256_broadcast_ss(kernel);
                __m256 result = _mm256_mul_ps(pixels, kernel_val);

                for (unsigned int k = 1; k <= const_radius(radius); ++k)
                {
                    kernel_val = _mm256_broadcast_ss(kernel + k);
                    __m256 pixel_right = (y + k < height)
                        ? _mm256_maskload_ps(in + (y + k) * in_stride + x, mask)
                        : _mm256_maskload_ps(
                            in + (height - ((k + y) % height) - 2) * in_stride + x, mask);

                    __m256 pixel_left = (k > y)
                        ? _mm256_maskload_ps(in + (k - y) * in_stride + x, mask)
                        : _mm256_maskload_ps(in + (y - k) * in_stride + x, mask);

                    pixels = addsub(pixel_right, pixel_left);
                    result = _mm256_fmadd_ps(pixels, kernel_val, result);
                }

                _mm256_store_ps(tmpptr + x, result);
            }

            const unsigned writeidx = (y + 1) % (const_radius(radius) + 1);
            float *writeptr = tmp + writeidx * n_outer_aligned;
            memcpy(out + (y - const_radius(radius)) * out_stride, writeptr, width * sizeof(float));
        }

        // copy from scratch memory to real output
        for (unsigned k = 0; k < const_radius(radius); ++k)
        {
            unsigned y = height + k;
            const unsigned writeidx = (y + 1) % (const_radius(radius) + 1);
            float *writeptr = tmp + writeidx * n_outer_aligned;
            memcpy(out + (y - const_radius(radius)) * out_stride, writeptr, width * sizeof(float));
        }

        fastfilters_memory_align_free(tmp);
    }
};

//@}

} // namespace vigra

#endif // VIGRA2_FASTFILTERS_HXX
