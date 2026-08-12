#include "lima/scheduling/solver.hpp"

#include "lima/scheduling/ida_star.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>

namespace lima {
namespace {

// Deliberately simple relocation heuristic used as the suboptimal comparison
// point (experiment E10) and as a fallback-policy candidate (E2).  Complete-
// ness is NOT guaranteed; failures return nullopt with outcome "no_solution".
class GreedySolver final : public StackSolver {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "greedy"; }

    [[nodiscard]] std::optional<std::vector<StackMove>> solve(
        const StackProblem& problem, SolverStats* const stats) override {
        SolverStats local;
        SolverStats& out = stats ? *stats : local;
        const auto started = std::chrono::steady_clock::now();
        const auto finish = [&](const std::string_view outcome) {
            out.outcome = outcome;
            out.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        };

        const std::size_t count = problem.stacks.size();
        if (count == 0 || count > 4 || problem.capacities.size() != count) {
            finish("rejected");
            return std::nullopt;
        }
        auto stacks = problem.stacks;
        const auto& caps = problem.capacities;
        std::size_t total = 0;
        std::array<int, 4> item_counts{};
        for (std::size_t s = 0; s < count; ++s) {
            if (caps[s] < 0 || stacks[s].size() > static_cast<std::size_t>(caps[s])) {
                finish("rejected");
                return std::nullopt;
            }
            total += stacks[s].size();
            for (const int item : stacks[s]) {
                if (item < 0 || static_cast<std::size_t>(item) >= count) {
                    finish("rejected");
                    return std::nullopt;
                }
                ++item_counts[static_cast<std::size_t>(item)];
            }
        }
        std::array<bool, 4> overflow{};
        for (std::size_t s = 0; s < count; ++s) overflow[s] = item_counts[s] > caps[s];

        const auto stack_done = [&](const std::size_t s) {
            std::size_t p = 0;
            if (overflow[s]) {
                if (stacks[s].size() != static_cast<std::size_t>(caps[s])) return false;
                while (p < stacks[s].size() && stacks[s][p] == static_cast<int>(s)) ++p;
                return p == stacks[s].size();
            }
            while (p < stacks[s].size() && stacks[s][p] == static_cast<int>(s)) ++p;
            while (p < stacks[s].size() && overflow[static_cast<std::size_t>(stacks[s][p])]) ++p;
            return p == stacks[s].size();
        };
        const auto all_done = [&]() {
            for (std::size_t s = 0; s < count; ++s) if (!stack_done(s)) return false;
            return true;
        };
        const auto target_ready = [&](const std::size_t t) {
            for (const int item : stacks[t]) if (item != static_cast<int>(t)) return false;
            return true;
        };

        std::vector<StackMove> moves;
        StackMove last{-1, -1};
        const std::size_t limit = 4 * total * total + 16;
        for (std::size_t step = 0; step < limit; ++step) {
            ++out.expanded_nodes;
            if (all_done()) {
                out.solution_length = moves.size();
                finish("solved");
                return moves;
            }
            int src = -1;
            int dst = -1;
            // 1. Direct placement: a top item whose target stack is clean and open.
            for (std::size_t s = 0; s < count && src < 0; ++s) {
                if (stacks[s].empty()) continue;
                const int item = stacks[s].back();
                const auto t = static_cast<std::size_t>(item);
                if (t != s && !overflow[t] && target_ready(t)
                    && stacks[t].size() < static_cast<std::size_t>(caps[t])
                    && !(static_cast<int>(s) == last.second && item == last.first)) {
                    src = static_cast<int>(s);
                    dst = item;
                }
            }
            // 2. Unearth: pop the top of the most-buried unfinished stack onto the
            //    stack with the most free space.
            if (src < 0) {
                int deepest = -1;
                for (std::size_t s = 0; s < count; ++s) {
                    if (stacks[s].empty() || stack_done(s)) continue;
                    const int depth = static_cast<int>(stacks[s].size());
                    if (depth > deepest) {
                        deepest = depth;
                        src = static_cast<int>(s);
                    }
                }
                if (src >= 0) {
                    int best_slack = 0;
                    for (std::size_t d = 0; d < count; ++d) {
                        if (static_cast<int>(d) == src) continue;
                        if (static_cast<int>(d) == last.first && src == last.second) continue;
                        const int slack = caps[d] - static_cast<int>(stacks[d].size());
                        if (slack > best_slack) {
                            best_slack = slack;
                            dst = static_cast<int>(d);
                        }
                    }
                }
            }
            if (src < 0 || dst < 0) {
                finish("no_solution");
                return std::nullopt;
            }
            const int item = stacks[static_cast<std::size_t>(src)].back();
            stacks[static_cast<std::size_t>(src)].pop_back();
            stacks[static_cast<std::size_t>(dst)].push_back(item);
            moves.emplace_back(src, dst);
            last = {src, dst};
        }
        finish("iteration_limit");
        return std::nullopt;
    }
};

}  // namespace

std::unique_ptr<StackSolver> make_solver(const SolverConfig& config) {
    if (config.kind == "ida") {
        return std::make_unique<IdaStarSolver>(IdaStarOptions{
            config.max_iterations, config.greedy_fastpath, config.bound_step});
    }
    if (config.kind == "greedy") {
        return std::make_unique<GreedySolver>();
    }
    throw std::invalid_argument("unknown solver kind: " + config.kind);
}

}  // namespace lima
