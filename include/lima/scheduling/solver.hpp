#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lima {

// A relocation: pop the top item of stack `first`, push it onto stack `second`.
using StackMove = std::pair<int, int>;

// A column-labeled stack rearrangement instance.  Stack entries are target
// stack indices ordered bottom-to-top; `capacities` bounds each stack height.
struct StackProblem {
    std::vector<std::vector<int>> stacks;
    std::vector<int> capacities;
};

// Per-invocation telemetry.  Filled by every solver so experiment harnesses
// can log solve-time distributions without knowing the solver internals.
struct SolverStats {
    std::uint64_t iterations{};        // threshold restarts (IDA*) or passes (greedy)
    std::uint64_t expanded_nodes{};    // search-tree node visits / greedy move trials
    double wall_seconds{};
    std::size_t solution_length{};
    bool fastpath_solved{};            // solved by the deterministic greedy fast path
    std::string_view outcome{"unsolved"};  // solved | no_solution | iteration_limit | rejected
};

// Strategy interface for the local intersection solver.  Implementations must
// be deterministic for a fixed problem and configuration.
class StackSolver {
public:
    virtual ~StackSolver() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::optional<std::vector<StackMove>> solve(
        const StackProblem& problem, SolverStats* stats = nullptr) = 0;
};

struct SolverConfig {
    std::string kind{"ida"};             // ida | greedy
    std::size_t max_iterations{1'000'000};
    // Scaled (x2) threshold increment between IDA* iterations.  The shipped
    // default 6 trades optimality for speed; 0 restores the textbook next-bound
    // schedule (optimal solutions, more re-expansions).
    int bound_step{6};
    bool greedy_fastpath{true};          // try the single-move fast path first (IDA*)
    int max_capacity{16};                // baseline stack-capacity acceptance bound (storage limit 64)
};

[[nodiscard]] std::unique_ptr<StackSolver> make_solver(const SolverConfig& config);

}  // namespace lima
