#pragma once

#include <math.h>
#include <fftw3.h>
#include <vector>
#include "problem.hpp"
#include "vector_handler.hpp"

class AbstractSolver
{
protected:
    VectorHandler vh;

public:
    virtual Result solve(const Problem& p) = 0;

    virtual ~AbstractSolver() {}

protected:
    /* init new solving process */
    virtual void init(const Problem& p) = 0;

    /* dispose resources after solving process finish */
    virtual void clear() = 0;
};

/* solves nonlinear equilibrium problem */
class NeumanSolver : public AbstractSolver
{
public:
    Result solve(const Problem& p) override;

private:
    double N;                     /* first moment */
    std::vector<double> C;        /* second moment samples */

    std::vector<double> m;        /* birth kernel samples */
    std::vector<double> w;        /* death kernel samples */

    std::vector<double> mC;       /* samples of [m * C] */
    std::vector<double> wC;       /* samples of [w * C] */
    std::vector<double> CwC;      /* samples of [Cw * C] */
    std::vector<double> w_mult_C; /* samples of wC */

    void init(const Problem& p) override;

    void clear() override;

    /* get zero padded samples of birth and death kernels and for C */
    void getVectors(const Problem& p);


    fftw_complex* tmp_C;    /* variables for holding tmp results of */
    fftw_complex* tmp_wC;   /* convolutions */
    fftw_complex* tmp_back;

    fftw_complex* fft_m;    /* fft of birth kernel */
    fftw_complex* fft_w;    /* fft of death kernel */

    fftw_plan forward_C;    /* plans for fftw3 */
    fftw_plan forward_wC;
    fftw_plan backward_mC;
    fftw_plan backward_wC;
    fftw_plan backward_CwC;

    void initConvolving(const Problem& p);

    void clearConvolving();

    void getConvolutions(const Problem& p);

    /* compute FFT of birth and death kernel */
    void getMWFFT(const Problem& p);

    /* convolving function */
    void convolve(const fftw_complex* f, const fftw_complex* g,
                  const fftw_plan& plan, std::vector<double>& res, const Problem& p);
};


class LinearNeumanSolver : public AbstractSolver
{
public:
    Result solve(const Problem& p) override;

private:
    fftw_complex* fft_C;    /* fft of the second moment */
    fftw_complex* fft_m;    /* fft of birth kernel */

    fftw_plan forward_C;    /* plans for fftw3 */
    fftw_plan backward_mC;

    std::vector<double> m;              /* samples of birth kernel */
    std::vector<double> w;              /* samples of death kernel */
    std::vector<double> C;              /* samples of the second moment */
    std::vector<double> mC;             /* samples of [m * C] */

    void init(const Problem& p) override;

    void clear() override;

    /* inits convolving */
    void initConvolving(const Problem& p);

    /* convolves birth kernel with the second moment */
    void convolve(const Problem& p);

    /* solves twin equation with the given parameter */
    void solveTwin(const Problem& p, double N);
};
