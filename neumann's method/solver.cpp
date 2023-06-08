#include "solver.hpp"

Result NeumanSolver::solve(const Problem& p)
{
    Result res;
    init(p);

    for (int i = 0; i < p.iters(); i++)
    {
        N = (p.b() - p.d()) / (vh.getDot(C, w, p.nodes(), p.step(), p.origin()) + p.s());
        getConvolutions(p);

        for (int j = 0; j < p.nodes(); j++)
        {
            C[j] = (m[j] / N - w[j] + mC[j] + p.b() - p.d() -
                    N / (p.alpha() + p.gamma()) *
                    (p.alpha() * (p.b() - p.d()) / N +
                     p.beta() * (wC[j] + CwC[j]) +
                     p.gamma() * ((p.b() - p.d()) / N + wC[j] + CwC[j]))) /
                   (p.d() + w[j] + N / (p.alpha() + p.gamma()) *
                                   p.alpha() * (p.b() - p.d()) / N + p.beta() * p.s());
        }
    }

    /* correcting second moment */
    double NN = N * N;
    for (int i = 0; i < p.nodes(); i++)
    {
        C[i] = NN * (C[i] + 1);
    }

    res.N = N;
    res.C.assign(p.nodes(), 0);
    res.dim = p.dimension();
    res.n_count = p.nodes();
    vh.copy(res.C, C, p.nodes());

    clear();
    return res;
}

void NeumanSolver::init(const Problem& p)
{
    getVectors(p);
    initConvolving(p);
}

void NeumanSolver::getVectors(const Problem& p)
{
    vh = VectorHandler(p.dimension());

    w.assign(p.nodes() * 2, 0);
    m.assign(p.nodes() * 2, 0);
    C.assign(p.nodes() * 2, 0);

    mC.assign(p.nodes() * 2, 0);
    wC.assign(p.nodes() * 2, 0);
    CwC.assign(p.nodes() * 2, 0);
    w_mult_C.assign(p.nodes() * 2, 0);

    double x = p.origin();

    for (int i = 0; i < p.nodes(); i++)
    {
        m[i] = p.getKernels().m(x);
        w[i] = p.getKernels().w(x);
        x += p.step();
    }

    double nm = vh.getIntNorm(m, p.nodes(), p.step(), p.origin());
    double nw = vh.getIntNorm(w, p.nodes(), p.step(), p.origin());

    for (int i = 0; i < p.nodes(); i++)
    {
        m[i] *= p.b() / nm;
        C[i] = w[i] = w[i] * p.s() / nw;
    }

    for (int i = p.nodes(); i < 2 * p.nodes(); i++)
    {
        m[i] = w[i] = w_mult_C[i] = C[i] = wC[i] = mC[i] = CwC[i] = 0.0;
    }
}

void NeumanSolver::clear()
{
    clearConvolving();
}

void NeumanSolver::initConvolving(const Problem& p)
{
    int n = p.nodes();

    tmp_C = fftw_alloc_complex(n + 1);
    tmp_wC = fftw_alloc_complex(n + 1);
    tmp_back = fftw_alloc_complex(n + 1);
    fft_m = fftw_alloc_complex(n + 1);
    fft_w = fftw_alloc_complex(n + 1);

    forward_C = fftw_plan_dft_r2c_1d(n * 2, C.data(), tmp_C,
                                     FFTW_ESTIMATE);
    forward_wC = fftw_plan_dft_r2c_1d(n * 2, w_mult_C.data(), tmp_wC,
                                      FFTW_ESTIMATE);
    backward_mC = fftw_plan_dft_c2r_1d(n * 2, tmp_back, mC.data(),
                                       FFTW_ESTIMATE);
    backward_wC = fftw_plan_dft_c2r_1d(n * 2, tmp_back, wC.data(),
                                       FFTW_ESTIMATE);
    backward_CwC = fftw_plan_dft_c2r_1d(n * 2, tmp_back, CwC.data(),
                                        FFTW_ESTIMATE);

    getMWFFT(p);
}

void NeumanSolver::getMWFFT(const Problem& p)
{
    std::vector<double> tmp_m;
    std::vector<double> tmp_w;

    fftw_plan m_plan;
    fftw_plan w_plan;

    if (p.dimension() == 3)
    {
        tmp_m.assign(2 * p.nodes(), 0);
        tmp_w.assign(2 * p.nodes(), 0);
        double x = p.origin();

        for (int i = 0; i < 2 * p.nodes(); i++)
        {
            tmp_m[i] = 4 * M_PI * fabs(x) * m[i];
            tmp_w[i] = 4 * M_PI * fabs(x) * w[i];
            x += p.step();
        }

        m_plan = fftw_plan_dft_r2c_1d(p.nodes() * 2, tmp_m.data(), fft_m,
                                      FFTW_ESTIMATE);
        w_plan = fftw_plan_dft_r2c_1d(p.nodes() * 2, tmp_w.data(), fft_w,
                                      FFTW_ESTIMATE);
    } else
    {
        m_plan = fftw_plan_dft_r2c_1d(p.nodes() * 2, m.data(), fft_m,
                                      FFTW_ESTIMATE);
        w_plan = fftw_plan_dft_r2c_1d(p.nodes() * 2, w.data(), fft_w,
                                      FFTW_ESTIMATE);
    }

    fftw_execute(m_plan);
    fftw_execute(w_plan);

    fftw_destroy_plan(m_plan);
    fftw_destroy_plan(w_plan);

}

void NeumanSolver::clearConvolving()
{
    fftw_free(tmp_C);
    fftw_free(tmp_wC);
    fftw_free(tmp_back);
    fftw_free(fft_m);
    fftw_free(fft_w);

    fftw_destroy_plan(forward_C);
    fftw_destroy_plan(forward_wC);
    fftw_destroy_plan(backward_mC);
    fftw_destroy_plan(backward_wC);
    fftw_destroy_plan(backward_CwC);

    fftw_cleanup();
}

void NeumanSolver::getConvolutions(const Problem& p)
{
    VectorHandler::multiplyVecs(C, w, w_mult_C, p.nodes());

    if (p.dimension() == 3)
    {
        double x = p.origin();

        for (int i = 0; i < p.nodes(); i++)
        {
            w_mult_C[i] *= 4 * M_PI * fabs(x);
            x += p.step();
        }
    }

    for (int i = p.nodes(); i < 2 * p.nodes(); i++)
    {
        w_mult_C[i] = C[i] = 0;
    }

    fftw_execute(forward_C);
    fftw_execute(forward_wC);

    convolve(tmp_C, fft_m, backward_mC, mC, p);
    convolve(tmp_C, fft_w, backward_wC, wC, p);
    convolve(tmp_wC, tmp_C, backward_CwC, CwC, p);

    VectorHandler::shiftLeft(mC, p.nodes(), p.nodes() / 2);
    VectorHandler::shiftLeft(wC, p.nodes(), p.nodes() / 2);
    VectorHandler::shiftLeft(CwC, p.nodes(), p.nodes() / 2);
}

void NeumanSolver::convolve(const fftw_complex* f, const fftw_complex* g,
                            const fftw_plan& plan, std::vector<double>& res, const Problem& p)
{
    for (int i = 0; i < p.nodes() + 1; i++)
    {
        tmp_back[i][0] = f[i][0] * g[i][0] - f[i][1] * g[i][1];
        tmp_back[i][1] = f[i][0] * g[i][1] + f[i][1] * g[i][0];
    }

    fftw_execute(plan);

    for (int i = 0; i < p.nodes() * 2; i++)
        res[i] *= p.step() / (p.nodes() * 2);
}


void LinearNeumanSolver::solveTwin(const Problem& p, double N)
{
    for (int i = 0; i < p.nodes(); i++)
    {
        C[i] = w[i];
    }

    for (int i = p.nodes(); i < 2 * p.nodes(); i++)
    {
        C[i] = 0.0;
    }

    for (int i = 0; i < p.iters(); i++)
    {
        convolve(p);
        for (int j = 0; j < p.nodes(); j++)
        {
            C[j] = (p.b() * mC[j] + m[j] * N - p.s() * (m[j] + w[j])) /
                   (p.b() + p.s() * w[j]);
        }
    }
}

Result LinearNeumanSolver::solve(const Problem& p)
{
    Result res;
    init(p);

    double N;

    solveTwin(p, 0);
    N = p.s() * vh.getDot(w, C, p.nodes(), p.step(), p.origin());

    solveTwin(p, 1);
    N = N / (1 - p.s() * vh.getDot(w, C, p.nodes(), p.step(), p.origin()));

    solveTwin(p, N);


    res.C.assign(p.nodes(), 0);
    res.dim = p.dimension();
    res.n_count = p.nodes();
    vh.copy(res.C, C, p.nodes());

    for (int i = 0; i < p.nodes(); i++)
    {
        res.C[i] += 1.0;
    }
    res.N = (p.b() - p.d()) /
            (p.s() * vh.getDot(res.C, w, p.nodes(), p.step(), p.origin()));

    clear();
    return res;
}


void LinearNeumanSolver::init(const Problem& p)
{
    vh = VectorHandler(p.dimension());

    m.assign(p.nodes() * 2, 0);
    w.assign(p.nodes(), 0);
    C.assign(p.nodes() * 2, 0);
    mC.assign(p.nodes() * 2, 0);

    double x = p.origin();

    for (int i = 0; i < p.nodes(); i++)
    {
        m[i] = p.getKernels().m(x);
        w[i] = p.getKernels().w(x);
        x += p.step();
    }

    double nm = vh.getIntNorm(m, p.nodes(), p.step(), p.origin());
    double nw = vh.getIntNorm(w, p.nodes(), p.step(), p.origin());

    for (int i = 0; i < p.nodes(); i++)
    {
        m[i] /= nm;
        w[i] /= nw;
    }

    for (int i = p.nodes(); i < 2 * p.nodes(); i++)
    {
        m[i] = C[i] = mC[i] = 0.0;
    }

    initConvolving(p);
}


void LinearNeumanSolver::initConvolving(const Problem& p)
{
    int n = p.nodes();

    fft_m = fftw_alloc_complex(n + 1);
    fft_C = fftw_alloc_complex(n + 1);

    forward_C = fftw_plan_dft_r2c_1d(n * 2, C.data(), fft_C,
                                     FFTW_ESTIMATE);
    backward_mC = fftw_plan_dft_c2r_1d(n * 2, fft_C, mC.data(),
                                       FFTW_ESTIMATE);

    fftw_plan m_plan = fftw_plan_dft_r2c_1d(p.nodes() * 2, m.data(), fft_m,
                                            FFTW_ESTIMATE);
    fftw_execute(m_plan);
    fftw_destroy_plan(m_plan);
}


void LinearNeumanSolver::clear()
{
    fftw_free(fft_m);
    fftw_free(fft_C);
    fftw_destroy_plan(forward_C);
    fftw_destroy_plan(backward_mC);
    fftw_cleanup();

}


void LinearNeumanSolver::convolve(const Problem& p)
{
    fftw_execute(forward_C);

    double re;
    double im;

    for (int i = 0; i < p.nodes() + 1; i++)
    {
        re = fft_C[i][0] * fft_m[i][0] - fft_C[i][1] * fft_m[i][1];
        im = fft_C[i][0] * fft_m[i][1] + fft_C[i][1] * fft_m[i][0];

        fft_C[i][0] = re;
        fft_C[i][1] = im;
    }

    fftw_execute(backward_mC);

    for (int i = 0; i < p.nodes() * 2; i++)
    {
        mC[i] *= p.step() / (p.nodes() * 2);
    }

    VectorHandler::shiftLeft(mC, p.nodes(), p.nodes() / 2);
}