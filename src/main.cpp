// src/main.cpp
// Command-line driver: nastran_solver [--cpu] <input.bdf> [output.f06]
//
// By default the Vulkan GPU backend is used when available; pass --cpu to
// force the Eigen CPU backend regardless.

#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#ifdef HAVE_VULKAN
#  include "solver/vulkan_solver_backend.hpp"
#endif
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>

static void print_usage() {
    std::cerr << "Usage: nastran_solver [--cpu] <input.bdf> [output.f06]\n"
              << "  --cpu   Force Eigen CPU solver (default: use Vulkan GPU if available)\n";
}

int main(int argc, char* argv[]) {
    // ── Argument parsing ─────────────────────────────────────────────────────
    bool force_cpu = false;
    int  positional = 0;
    std::filesystem::path bdf_path;
    std::filesystem::path f06_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--cpu") {
            force_cpu = true;
        } else if (positional == 0) {
            bdf_path = arg;
            ++positional;
        } else if (positional == 1) {
            f06_path = arg;
            ++positional;
        } else {
            std::cerr << "Unexpected argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (positional == 0) {
        print_usage();
        return 1;
    }
    if (f06_path.empty())
        f06_path = std::filesystem::path(bdf_path).replace_extension(".f06");

    try {
        auto t0 = std::chrono::steady_clock::now();

        std::cout << "Reading: " << bdf_path << "\n";
        nastran::Model model = nastran::BdfParser::parse_file(bdf_path);

        std::cout << "  Nodes:    " << model.nodes.size()             << "\n";
        std::cout << "  Elements: " << model.elements.size()          << "\n";
        std::cout << "  Materials:" << model.materials.size()         << "\n";
        std::cout << "  Subcases: " << model.analysis.subcases.size() << "\n";

        // ── Backend selection ─────────────────────────────────────────────────
        std::unique_ptr<nastran::SolverBackend> backend;

#ifdef HAVE_VULKAN
        if (!force_cpu) {
            auto vk = nastran::VulkanSolverBackend::try_create();
            if (vk.has_value()) {
                backend = std::make_unique<nastran::VulkanSolverBackend>(std::move(*vk));
            } else {
                std::cout << "Vulkan unavailable — falling back to Eigen CPU solver\n";
            }
        }
#endif
        if (!backend)
            backend = std::make_unique<nastran::EigenSolverBackend>();

        std::cout << "Solving with: " << backend->name() << "\n";
        nastran::LinearStaticSolver solver(std::move(backend));
        nastran::SolverResults results = solver.solve(model);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
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
