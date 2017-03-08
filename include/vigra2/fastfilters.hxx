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

template <KernelSymmetry SYMMETRY = KernelEven, ArrayIndex RADIUS = runtime_size>
struct AvxConvolveLine
{
    //static void exec_x(float * in, ArrayIndex size, float * out,
    //                   float * kernel)
    //{
    //    for (ArrayIndex x = 0; x < radius; ++x)
    //    {
    //        float sum = kernel->coefs[0] * cur_input[x];

    //        for (unsigned int k = 1; k < x; ++k) {
    //            float left;
    //            if (-(int)k + (int)x < 0)
    //                left = in_border_left[y * borderptr_outer_stride + (FF_KERNEL_LEN - (int)k + (int)x)];
    //            else
    //                left = cur_input[x - k];

    //            sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[x + k], left);
    //        }

    //        cur_output[x] = sum;
    //    }
    //}
};

template <KernelSymmetry SYMMETRY>
struct AvxConvolveLine<SYMMETRY, runtime_size>
{
    static void exec_x(float * in, ArrayIndex size, float * out,
                       float * kernel, ArrayIndex radius)
    {
        AddBySymmetry<SYMMETRY> addsub;

        // FIXME: check kernel longer than line
        ArrayIndex x = 0;
        for (; x < radius; ++x)
        {
            float sum = kernel[0] * in[x];

            for (ArrayIndex k = 1; k <= radius; ++k)
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
        ArrayIndex steps = (size - radius - x) / 32;
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
            for (ArrayIndex k = 1; k <= radius; ++k)
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
        steps = (size - radius - x) / 8;
        for (ArrayIndex j = 0; j < steps; ++j, x += 8)
        {
            // load next 8 pixels
            __m256 result0 = _mm256_loadu_ps(in + x);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_broadcast_ss(kernel);
            result0 = _mm256_mul_ps(result0, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            for (ArrayIndex k = 1; k <= radius; ++k)
            {
                kernel_val = _mm256_broadcast_ss(kernel + k);

                __m256 pixels0 = addsub(_mm256_loadu_ps(in + x + k), _mm256_loadu_ps(in + (x - k)));

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
            }
            _mm256_storeu_ps(out + x, result0);
        }

        // FIXME: check if this must be radius + 1
        for (; x < size - radius; ++x)
        {
            float sum = kernel[0] * in[x];

            // work on both sides of symmetric kernel simultaneously
            for (ArrayIndex k = 1; k <= radius; ++k)
            {
                sum += kernel[k] * addsub(in[x + k], in[x - k]);
            }
            out[x] = sum;
        }

        // right border
        for (; x < size; ++x)
        {
            float sum = in[x] * kernel[0];

            for (ArrayIndex k = 1; k <= radius; ++k)
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
};

//@}

} // namespace vigra

#endif // VIGRA2_FASTFILTERS_HXX
