#include <stdio.h>
#include <iostream>
#include "solver.hpp"
#include "problem.hpp"

int main(int argc, char** argv)
{
    Problem equation(argc, argv);

    AbstractSolver* solver;

    if (equation.getMethod() == "linear")
        solver = new LinearNeumanSolver;
    else
    {
        solver = new NeumanSolver;
        if (equation.getMethod() != "nonlinear")
            std::cout << "Unknown method, using nonlinear Neuman method" << std::endl;
    }

    Result answer = solver->solve(equation);

    printf("First moment: %.*lf\nC(0) = %.*lf\n",
           equation.accuracy(), answer.N, equation.accuracy(),
           answer.getC0());

    VectorHandler::storeVector(answer.C, equation.path().c_str(),
                               equation.nodes(), equation.step(), equation.origin(),
                               equation.accuracy());
    delete solver;
    return 0;
}
