#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include "kernels.hpp"
#include "vector_handler.hpp"

class Problem
{
public:

private:
    AbstractKernels* kernels;   /* birth and death parameters */

    double _b;          /* birth rate */
    double _s;          /* competition rate */
    double _d;          /* death rate */

    double _alpha;      /* closure parameters */
    double _beta;
    double _gamma;

    int i_count;        /* iteration count */
    int n_count;        /* nodes count */

    double orgn;        /* origin for integration */
    double _R;          /* integration limit */
    double _step;       /* step between nodes */
    int dim;            /* dimension of space */

    int acc;            /* output accurancy */
    std::string _path;  /* path for storing data */
    std::string method; /* method of solving */
    /* if _path is empty output file won't be created */

public:
    Problem(int argc, char** argv);

    ~Problem() { delete kernels; }

    /* getters */
    const AbstractKernels& getKernels() const { return *kernels; }

    double b() const { return _b; }

    double s() const { return _s; }

    double d() const { return _d; }

    double alpha() const { return _alpha; }

    double beta() const { return _beta; }

    double gamma() const { return _gamma; }

    int iters() const { return i_count; }

    int nodes() const { return n_count; }

    double origin() const { return orgn; }

    double R() const { return _R; }

    double step() const { return _step; }

    int dimension() const { return dim; }

    int accuracy() const { return acc; }

    std::string path() const { return _path; }

    std::string getMethod() const { return method; }

    /* some useful properties */
    double getDispersionM() const;

    double getDispersionW() const;

    double getExcessM() const;

    double getExcessW() const;
};

struct Result
{
    double N;
    int dim;
    size_t n_count;
    std::vector<double> C;

    Result() {}

    /* Second moment at the origin */
    double getC0() const
    {
        return C[0];
    }
};
