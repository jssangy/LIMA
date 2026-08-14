#pragma once

#include "lima/scheduling/ida_star.hpp"
#include "lima/scheduling/solver.hpp"

#include <cstddef>

namespace lima {

// Best-first graph-search families used in the Gate A local-solver
// tournament.  A* is exact when paired with an admissible bf/tt lower bound;
// the other modes intentionally trade optimality for search effort.
enum class BestFirstMode { kAStar, kWeightedAStar, kGreedyBestFirst, kUniformCost };

struct BestFirstOptions {
    BestFirstMode mode{BestFirstMode::kAStar};
    IdaLbMode lb_mode{IdaLbMode::kTt};
    double heuristic_weight{2.0};
    std::size_t max_expanded_nodes{2'000'000};
    int max_capacity{16};
};

class BestFirstSolver final : public StackSolver {
public:
    explicit BestFirstSolver(BestFirstOptions options = {}) : options_(options) {}
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::optional<std::vector<StackMove>> solve(
        const StackProblem& problem, SolverStats* stats = nullptr) override;

private:
    BestFirstOptions options_;
};

}  // namespace lima
