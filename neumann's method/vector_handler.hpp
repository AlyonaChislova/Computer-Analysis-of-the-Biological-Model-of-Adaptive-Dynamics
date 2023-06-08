#pragma once

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <vector>

class VectorHandler
{
    int dim;
    double dimCoeff;

public:
    /* multiply two vectors componentwise and save result in the third
     * one */
    static void multiplyVecs(const std::vector<double>& f, const std::vector<double>& g, std::vector<double>& fg,
                             int size);

    /* shift vector left (not rolling) */
    static void shiftLeft(std::vector<double>& f, int size, int shft);

    /* copies vector to another storage */
    static void copy(std::vector<double>& dst, const std::vector<double>& src, int count);

    /* store vector into the file as list of pair 'x f(x)' */
    static void storeVector(const std::vector<double>& f, const char* path, int size,
                            double step, double origin, int accuracy);

    static double weight(int i, int n, double step)
    {
        return i == 0 || i == n - 1 ? step / 2 : step;
    }

    VectorHandler() : dim(1) { dimCoeff = 2.0; }

    VectorHandler(int dim) : dim(dim)
    {
        dimCoeff = dim == 1 ? 2.0 : dim == 2 ? 2.0 * M_PI : 4.0 * M_PI;
    }

    /* get integral dot product */
    double getDot(const std::vector<double>& f, const std::vector<double>& g, int size, double step,
                  double origin) const;

    /* return integral norm of vecor */
    double getIntNorm(const std::vector<double>& f, int size, double step,
                      double origin) const;


private:
    /* TODO: make jacobian more logical and faster */
    double jacobian(double x) const
    {
        return
                dim == 1 ? 1.0 :
                dim == 2 ? x :
                x * x;
    }
};

