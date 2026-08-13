#pragma once

#include "lima/scheduling/solver.hpp"

#include <cstddef>
#include <vector>

namespace lima {

struct BeamSearchOptions {
    std::size_t beam_width{2'048};
    std::size_t max_depth{512};
    std::size_t max_expanded_nodes{1'000'000};
};

// Width-limited best-first search over stack rearrangement, exposed through
// the StackSolver strategy interface.  Deterministic for a fixed problem and
// options; suboptimal by construction (the beam may prune the optimal line).
class BeamSolver final : public StackSolver {
public:
    explicit BeamSolver(BeamSearchOptions options = {}) : options_(options) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "beam"; }
    [[nodiscard]] std::optional<std::vector<StackMove>> solve(
        const StackProblem& problem, SolverStats* stats = nullptr) override;

private:
    BeamSearchOptions options_;
};

// Legacy convenience wrapper kept for tools and tests: throws on failure.
// Stack entries are target stack indices and are ordered bottom-to-top.
std::vector<StackMove> solve_stack_rearrangement_beam(
    const std::vector<std::vector<int>>& stacks,
    const std::vector<int>& capacities,
    BeamSearchOptions options = {});

}  // namespace lima
