#include "problem.hpp"

#include <vector>
#include <argparse/argparse.hpp>

Problem::Problem(int argc, char** argv)
{
    argparse::ArgumentParser program("exec");

    program.add_argument("--kernels_type").default_value("norm")
            .help("Set kernels type:\n"\
                  "\tnorm : Gausian kernels\n"\
                  "\tkurtic : Equal kurtic kernels\n"\
                  "\tg_kurtic : General kurtic kernels\n"\
                  "\texp : Danchenko's exponential kernels\n"\
                  "\troughgarden : Roughgarden kernels\n"\
                  "\tpoly : Exponent polynomial kernels\n");

    program.add_argument("--kernels").default_value(std::vector < double > {1.0, 1.0})
            .help("Kernels parameters:\n"\
                  "\tGausian kernels : The standart deviations for the death and birth kernels\n"\
                  "\tEqual kurtic kernels : s0m = s0w and s1m = s1w (two numbers)\n"\
                  "\tGeneral kurtic kernels : s0m, s1m, s0w, and s1w\n"\
                  "\tDanchenko's exponential kernels : A and B\n"\
                  "\tRoughgarden kernels : sm, gamma_m, sw, and gamma_w\n"\
                  "\tExponent polynomial kernels : am, bm, aw, and bw\n")
            .nargs(2, 4)
            .scan<'g', double>();

    program.add_argument("-b").default_value(1.0)
            .help("Birth rating")
            .scan<'g', double>();
    program.add_argument("-s").default_value(1.0)
            .help("Competition rating (d')")
            .scan<'g', double>();
    program.add_argument("-d").default_value(0.0)
            .help("Environment death rating")
            .scan<'g', double>();

    program.add_argument("-A", "--alpha").default_value(1.0)
            .help("Alpha parameter of the closure")
            .scan<'g', double>();
    program.add_argument("-B", "--beta").default_value(1.0)
            .help("Beta parameter of the closure")
            .scan<'g', double>();
    program.add_argument("-G", "--gamma").default_value(1.0)
            .help("Gamma parameter of the closure")
            .scan<'g', double>();

    program.add_argument("-i", "--i_count").default_value(1000)
            .help("Iteration count")
            .scan<'i', int>();
    program.add_argument("-n", "--n_count").default_value(5000)
            .default_value(5000)
            .help("Nodes count")
            .scan<'i', int>();

    program.add_argument("-r").default_value(-1.0)
            .help("Integration area size (segment length, round or ball radius in 1D, 2D and 3D respectively)."\
                  "Don't set to calculate it automatically")
            .scan<'g', double>();
    program.add_argument("-D", "--dimension").default_value(1)
            .help("Environment dimesionality")
            .scan<'i', int>();
    program.add_argument("-e", "--accuracy").default_value(5)
            .help("Accurancy of the output in the signs after point")
            .scan<'i', int>();
    program.add_argument("-p", "--path").default_value(std::string{"graph.plt"})
            .help("Path to store calculated second moment");
    program.add_argument("--method").default_value(std::string("nonlinear"))
            .help("nonlinear : Non linear Neuman method\n"\
                  "linear : Linear Neuman method\n");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto kernels_type = program.get("--kernels_type");
    auto kernels_args = program.get < std::vector < double >> ("--kernels");
    if (kernels_type == "norm")
        kernels = new NormalKernels(kernels_args[0], kernels_args[1]);
    else if (kernels_type == "kurtic")
        kernels = new KurticKernels(kernels_args[0], kernels_args[1]);
    else if (kernels_type == "gen_kurtic")
        kernels = new KurticKernels(kernels_args[0], kernels_args[1], kernels_args[2], kernels_args[3]);
    else if (kernels_type == "exp")
        kernels = new ExponentKernels(kernels_args[0], kernels_args[1]);
    else if (kernels_type == "roughgarden")
        kernels = new RoughgardenKernels(kernels_args[0], kernels_args[1], kernels_args[2], kernels_args[3]);
    else if (kernels_type == "poly")
        kernels = new ExponentPolynomialKernels(kernels_args[0], kernels_args[1], kernels_args[2], kernels_args[3]);
    else
    {
        std::cerr << "Unkown kernels_type" << std::endl;
        std::exit(1);
    }

    _b = program.get<double>("-b");
    _s = program.get<double>("-s");
    _d = program.get<double>("-d");

    _alpha = program.get<double>("-A");
    _beta = program.get<double>("-B");
    _gamma = program.get<double>("-G");

    i_count = program.get<int>("-i");
    n_count = program.get<int>("-n");

    _R = program.get<double>("-r");
    dim = program.get<int>("-D");

    acc = program.get<int>("-e");
    _path = program.get("-p");
    method = program.get("--method");

    if (_R < 0)
    {
        _R = 0;
        while (std::abs(kernels->m(_R)) > 1e-9 || std::abs(kernels->w(_R)) > 1e-9)
            _R += 1e-5;

    }
    _step = _R / (n_count - 1);
    orgn = 0.0;
}
