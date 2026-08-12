#pragma once

#include "lima/scheduling/solver.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace lima::bench {

// Synthetic single-intersection CPMP instances for solver stress studies
// (experiment E1-B): N items with uniform-random target arms are packed into
// random slots across the given arm capacities, then handed to the solver.
struct BenchOptions {
    std::vector<int> capacities{10, 10, 10, 10};
    int items{8};
    int instances{100};
    std::uint64_t seed{0};
    std::string csv_path;
};

inline int run(StackSolver& solver, const BenchOptions& options) {
    std::ofstream csv;
    std::ostream* out = &std::cout;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path, std::ios::trunc);
        if (!csv) {
            std::cerr << "lima: cannot open bench csv: " << options.csv_path << "\n";
            return 1;
        }
        out = &csv;
    }
    *out << "instance,n,arms,outcome,iterations,expanded,wall_us,solution_len,fastpath\n";
    std::mt19937_64 rng(options.seed);
    const auto arm_count = options.capacities.size();
    std::string arms_text;
    for (std::size_t i = 0; i < arm_count; ++i) {
        if (i) arms_text += "/";
        arms_text += std::to_string(options.capacities[i]);
    }
    std::uint64_t failures = 0;
    for (int instance = 0; instance < options.instances; ++instance) {
        // Draw item slots without replacement over the flattened capacity.
        std::vector<std::pair<int, int>> slots;
        for (std::size_t s = 0; s < arm_count; ++s)
            for (int p = 0; p < options.capacities[s]; ++p) slots.emplace_back(static_cast<int>(s), p);
        std::shuffle(slots.begin(), slots.end(), rng);
        std::vector<std::vector<int>> stacks(arm_count);
        std::vector<std::vector<std::pair<int, int>>> chosen(arm_count);
        for (int k = 0; k < options.items && k < static_cast<int>(slots.size()); ++k)
            chosen[static_cast<std::size_t>(slots[static_cast<std::size_t>(k)].first)].push_back(slots[static_cast<std::size_t>(k)]);
        std::uniform_int_distribution<int> pick_target(0, static_cast<int>(arm_count) - 1);
        for (std::size_t s = 0; s < arm_count; ++s) {
            // compact occupied slots to the bottom, preserving draw order
            for (std::size_t k = 0; k < chosen[s].size(); ++k) stacks[s].push_back(pick_target(rng));
        }
        SolverStats stats;
        const auto result = solver.solve({stacks, options.capacities}, &stats);
        if (!result) ++failures;
        *out << instance << "," << options.items << "," << arms_text << "," << stats.outcome << ","
             << stats.iterations << "," << stats.expanded_nodes << ","
             << static_cast<std::uint64_t>(stats.wall_seconds * 1e6) << ","
             << stats.solution_length << "," << (stats.fastpath_solved ? 1 : 0) << "\n";
    }
    std::cerr << "bench: " << options.instances << " instances, " << failures << " unsolved\n";
    return 0;
}

}  // namespace lima::bench
