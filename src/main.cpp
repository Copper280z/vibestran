// src/main.cpp
// Command-line driver: nastran_solver <input.bdf> [output.f06]

#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#include <filesystem>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: nastran_solver <input.bdf> [output.f06]\n";
        return 1;
    }

    std::filesystem::path bdf_path(argv[1]);
    std::filesystem::path f06_path = (argc >= 3)
        ? std::filesystem::path(argv[2])
        : bdf_path.replace_extension(".f06");

    try {
        auto t0 = std::chrono::steady_clock::now();

        std::cout << "Reading: " << bdf_path << "\n";
        nastran::Model model = nastran::BdfParser::parse_file(bdf_path);

        std::cout << "  Nodes:    " << model.nodes.size()    << "\n";
        std::cout << "  Elements: " << model.elements.size() << "\n";
        std::cout << "  Materials:" << model.materials.size()<< "\n";
        std::cout << "  Subcases: " << model.analysis.subcases.size() << "\n";

        nastran::LinearStaticSolver solver(
            std::make_unique<nastran::EigenSolverBackend>());

        std::cout << "Solving...\n";
        nastran::SolverResults results = solver.solve(model);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1-t0).count();
        std::cout << "Solution complete in " << elapsed << " s\n";

        nastran::F06Writer::write(results, model, f06_path);
        std::cout << "Results written: " << f06_path << "\n";

    } catch (const nastran::ParseError& e) {
        std::cerr << "Parse error: " << e.what() << "\n";
        return 2;
    } catch (const nastran::SolverError& e) {
        std::cerr << "Solver error: " << e.what() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 4;
    }
    return 0;
}
