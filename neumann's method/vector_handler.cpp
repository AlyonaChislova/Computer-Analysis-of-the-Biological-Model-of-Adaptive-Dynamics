#include "vector_handler.hpp"

double VectorHandler::getDot(const std::vector<double>& f, const std::vector<double>& g, int size,
                             double step, double origin) const
{
    double res = 0.0;
    double x = origin;

    for (int i = 0; i < size; i++)
    {
        res += f[i] * g[i] * weight(i, size, step) * jacobian(x);
        x += step;
    }

    return res * dimCoeff;
}


double VectorHandler::getIntNorm(const std::vector<double>& f, int size, double step,
                                 double origin) const
{
    double res = 0.0;
    double x = origin;

    for (int i = 0; i < size; i++)
    {
        res += fabs(f[i]) * weight(i, size, step) * jacobian(x);
        x += step;
    }

    return res * dimCoeff;
}


void VectorHandler::multiplyVecs(const std::vector<double>& f, const std::vector<double>& g,
                                 std::vector<double>& fg, int size)
{
    for (int i = 0; i < size; i++)
    {
        fg[i] = f[i] * g[i];
    }
}


void VectorHandler::storeVector(const std::vector<double>& f, const char* path,
                                int size, double step, double origin, int accuracy)
{
    FILE* out = fopen(path, "w");
    double x = origin;

    for (int i = 0; i < size; i++)
    {
        fprintf(out,
                "%.*lf %.*lf\n",
                accuracy,
                x,
                accuracy,
                f[i]
        );
        x += step;
    }

    fclose(out);
}


void VectorHandler::shiftLeft(std::vector<double>& f, int size, int shft)
{
    for (int i = 0; i < size; i++)
    {
        f[i] = f[i + shft];
    }
}


void VectorHandler::copy(std::vector<double>& dst, const std::vector<double>& src, int count)
{
    for (int i = 0; i < count; i++)
    {
        dst[i] = src[i];
    }
}

