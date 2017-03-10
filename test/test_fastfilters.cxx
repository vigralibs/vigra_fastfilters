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

#include <typeinfo>
#include <iostream>
#include <vector>
#include <cmath>
#include <vigra2/unittest.hxx>
#include <vigra2/fastfilters.hxx>

using namespace vigra;

struct FastFiltersTest
{
    void testXFilter()
    {
        int size = 21;
        std::vector<float> data(size, 0.0f), res(size, 0.0f);
        data[size / 2] = 1.0f;

        float kernel[] = { 0.4f, 0.25f, 0.05f };
        AvxConvolveLine<KernelEven, 2>::exec_x(&data[0], size, &res[0], kernel, 2);

        for (int k = 0; k < size; ++k)
        {
            int d = std::abs(size / 2 - k);
            switch (d)
            {
            case 0:
            case 1:
            case 2:
                shouldEqual(res[k], kernel[d]);
                break;
            default:
                shouldEqual(res[k], 0.0f);
            }
        }
    }

    void testYFilter()
    {
        int size = 21;
        std::vector<float> data(size, 0.0f), res(size, 0.0f);
        data[size / 2] = 1.0f;

        float kernel[] = { 0.4f, 0.25f, 0.05f };
        AvxConvolveLine<KernelEven, 2>::exec_y(&data[0], 1, size, 1, &res[0], 1, kernel, 2);

        for (int k = 0; k < size; ++k)
        {
            int d = std::abs(size / 2 - k);
            switch (d)
            {
            case 0:
            case 1:
            case 2:
                shouldEqual(res[k], kernel[d]);
                break;
            default:
                shouldEqual(res[k], 0.0f);
            }
        }
    }

};


struct FastFiltersTestSuite
: public test_suite
{
    FastFiltersTestSuite()
    : test_suite("FastFiltersTestSuite")
    {
        add(testCase(&FastFiltersTest::testXFilter));
        add(testCase(&FastFiltersTest::testYFilter));
    }
};


int main(int argc, char ** argv)
{
    FastFiltersTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
