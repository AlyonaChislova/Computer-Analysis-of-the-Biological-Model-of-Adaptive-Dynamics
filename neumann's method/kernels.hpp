#pragma once

#include <cmath>

class AbstractKernels
{
public:
    virtual ~AbstractKernels() {}

    virtual double m(double x) const = 0;

    virtual double w(double x) const = 0;
};


class KurticKernels : public AbstractKernels
{
    double s0m;
    double s1m;
    double s0w;
    double s1w;

public:
    KurticKernels(double s0, double s1)
            : s0m(s0), s1m(s1), s0w(s0), s1w(s1) {}

    KurticKernels(double s0m, double s1m, double s0w, double s1w)
            : s0m(s0m), s1m(s1m), s0w(s0w), s1w(s1w) {}

    double m(double x) const
    {
        double xx = x * x;
        return std::exp(-0.5 * (s0m * xx + s1m * xx * xx) / (1 + xx));
    }

    double w(double x) const
    {
        double xx = x * x;
        return std::exp(-0.5 * (s0w * xx + s1w * xx * xx) / (1 + xx));
    }
};


class NormalKernels : public AbstractKernels
{
    static constexpr double gauss_coeff = 1.0 / sqrt(2 * M_PI);

    double sigma_m;
    double sigma_w;

public:
    NormalKernels(double sm, double sw) : sigma_m(sm), sigma_w(sw) {}

    double m(double x) const
    {
        double arg = x / sigma_m;
        return gauss_coeff / sigma_m * std::exp(-0.5 * arg * arg);
    }

    double w(double x) const
    {
        double arg = x / sigma_w;
        return gauss_coeff / sigma_w * std::exp(-0.5 * arg * arg);
    }
};


class ExponentKernels : public AbstractKernels
{
    double A;
    double B;

public:
    ExponentKernels(double A, double B) : A(A), B(B) {}

    double m(double x) const
    {
        return exp(-2 * fabs(x));
    }

    double w(double x) const
    {
        double ex = std::exp(-std::abs(x));
        double Axx = A * x * x;

        return (ex * (Axx / 3 - 16.0 / 9 * A * std::abs(x) + 56.0 / 27 * A + B / 3)) /
               (1 + ex * (Axx + B));
    }
};


class RoughgardenKernels : public AbstractKernels
{
    double sm;
    double gm;
    double sw;
    double gw;

public:
    RoughgardenKernels(double s, double g)
            : sm(s), gm(g), sw(s), gw(g) {}

    RoughgardenKernels(double sm, double gm, double sw, double gw)
            : sm(sm), gm(gm), sw(sw), gw(gw) {}

    double m(double x) const
    {
        return std::exp(-std::pow(std::abs(x / sm), gm));
    }

    double w(double x) const
    {
        return std::exp(-std::pow(std::abs(x / sw), gw));
    }
};


class ExponentPolynomialKernels : public AbstractKernels
{
    double am;
    double bm;
    double aw;
    double bw;

public:
    ExponentPolynomialKernels(double a, double b)
            : am(a), bm(b), aw(a), bw(b) {}

    ExponentPolynomialKernels(double am, double bm, double aw, double bw)
            : am(am), bm(bm), aw(aw), bw(bw) {}

    double m(double x) const
    {
        double xx = x * x;
        return std::exp(-am * xx - bm * xx * xx);
    }

    double w(double x) const
    {
        double xx = x * x;
        return std::exp(-aw * xx - bw * xx * xx);
    }
};
