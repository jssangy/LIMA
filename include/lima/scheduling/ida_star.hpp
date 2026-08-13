#pragma once

#include "lima/scheduling/solver.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace lima {

// Lower-bound family used by the IDA* threshold test (opt-in; the shipped
// default stays the legacy weighted heuristic).
//
//   kLegacy  the original hand-tuned estimate.  It is NOT admissible (it can
//            overestimate), so it prunes aggressively but forfeits optimality
//            even with bound_step == 0.
//   kBf      Bortfeldt & Forster (EJOR 217, 2012) style demand-supply bound
//            adapted to the relaxed sorted-with-overflow-parking goal used by
//            this solver: misplaced-item count plus forced re-entries derived
//            from per-type supply/demand imbalance across stacks.  Admissible
//            for the relaxed goal.  Optimality additionally requires
//            bound_step == 0 AND deterministic_move == false (the exclusive
//            greedy fast path can skip the optimal line); verified against a
//            brute-force BFS optimum on 400 random instances.
//   kTt      kBf plus a Tanaka & Tierney (EJOR 264, 2018) style refinement:
//            mutual cross-pair relocations between two home stacks force a
//            second move for the smaller side of each pair.  Also admissible.
enum class IdaLbMode { kLegacy, kBf, kTt };

struct IdaStarOptions {
    std::size_t max_iterations{1'000'000};
    bool deterministic_move{true};
    int bound_step{6};
    // Baseline acceptance bound: instances with a stack capacity above this
    // are rejected (nullopt).  Storage allows up to 64; raise explicitly for
    // long-corridor stress studies only.
    int max_capacity{16};
    IdaLbMode lb_mode{IdaLbMode::kLegacy};
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
