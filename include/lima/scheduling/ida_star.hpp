#pragma once

#include "lima/scheduling/solver.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace lima {

struct IdaStarOptions {
    std::size_t max_iterations{1'000'000};
    bool deterministic_move{true};
    int bound_step{6};
    // Baseline acceptance bound: instances with a stack capacity above this
    // are rejected (nullopt).  Storage allows up to 64; raise explicitly for
    // long-corridor stress studies only.
    int max_capacity{16};
};

class IdaStarSolver final : public StackSolver {
public:
    explicit IdaStarSolver(IdaStarOptions options = {}) : options_(options) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "ida"; }
    [[nodiscard]] std::optional<std::vector<StackMove>> solve(
        const StackProblem& problem, SolverStats* stats = nullptr) override;

private:
    IdaStarOptions options_;
};

// Legacy convenience wrapper kept for tools and tests: throws on failure.
std::vector<StackMove> solve_stack_rearrangement(
    const std::vector<std::vector<int>>& stacks,
    const std::vector<int>& capacities,
    IdaStarOptions options = {});

}  // namespace lima
