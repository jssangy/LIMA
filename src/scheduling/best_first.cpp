#include "lima/scheduling/best_first.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace lima {
namespace {

constexpr int kMaxStacks = 4;
constexpr int kMaxCapacity = 64;
constexpr int kEmpty = 16;

struct Key {
    std::uint64_t first{};
    std::uint64_t second{};
    friend bool operator==(const Key&, const Key&) = default;
};

struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
        std::uint64_t value = key.first
            ^ (key.second + 0x9e3779b97f4a7c15ULL + (key.first << 6U) + (key.first >> 2U));
        value ^= value >> 33U;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33U;
        return static_cast<std::size_t>(value);
    }
};

std::uint64_t splitmix64(std::uint64_t& value) {
    std::uint64_t z = (value += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

struct Zobrist {
    std::array<std::array<std::array<std::uint64_t, 17>, kMaxCapacity>, kMaxStacks> first{};
    std::array<std::array<std::array<std::uint64_t, 17>, kMaxCapacity>, kMaxStacks> second{};

    Zobrist() {
        std::uint64_t a = 0x6a09e667f3bcc909ULL;
        std::uint64_t b = 0xbb67ae8584caa73bULL;
        for (int s = 0; s < kMaxStacks; ++s)
            for (int p = 0; p < kMaxCapacity; ++p)
                for (int v = 0; v <= 16; ++v) {
                    first[s][p][v] = splitmix64(a);
                    second[s][p][v] = splitmix64(b);
                }
    }
};

const Zobrist kZobrist;

struct State {
    int count{};
    std::array<std::uint8_t, kMaxStacks> capacity{};
    std::array<std::uint8_t, kMaxStacks> size{};
    std::array<std::array<std::uint8_t, kMaxCapacity>, kMaxStacks> cells{};
    std::uint8_t overflow_mask{};
    Key hash{};
};

struct Node {
    State state;
    std::size_t parent{std::numeric_limits<std::size_t>::max()};
    StackMove move{-1, -1};
    int depth{};
};

struct QueueEntry {
    double priority{};
    int h2{};
    int depth{};
    std::size_t node{};
};

struct Worse {
    bool operator()(const QueueEntry& lhs, const QueueEntry& rhs) const noexcept {
        if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
        if (lhs.h2 != rhs.h2) return lhs.h2 > rhs.h2;
        if (lhs.depth != rhs.depth) return lhs.depth < rhs.depth;
        return lhs.node > rhs.node;
    }
};

bool solved(const State& state) {
    if (state.overflow_mask == 0) {
        for (int s = 0; s < state.count; ++s)
            for (int p = 0; p < state.size[s]; ++p)
                if (state.cells[s][p] != s) return false;
        return true;
    }
    for (int s = 0; s < state.count; ++s) {
        const bool overflow = (state.overflow_mask & (1U << s)) != 0;
        if (overflow) {
            if (state.size[s] != state.capacity[s]) return false;
            for (int p = 0; p < state.size[s]; ++p)
                if (state.cells[s][p] != s) return false;
        } else {
            int p = 0;
            while (p < state.size[s] && state.cells[s][p] == s) ++p;
            while (p < state.size[s]
                   && (state.overflow_mask & (1U << state.cells[s][p])) != 0) ++p;
            if (p != state.size[s]) return false;
        }
    }
    return true;
}

int legacy_h2(const State& state) {
    int value = 0;
    for (int s = 0; s < state.count; ++s) {
        bool clean = true;
        for (int p = 0; p < state.size[s]; ++p) {
            if (clean && state.cells[s][p] != s) {
                clean = false;
                value += (state.capacity[s] - p) * 3;
            } else if (!clean) {
                value += 2;
            }
        }
    }
    return value;
}

// Same admissible relaxed-goal lower bounds as IdaStarSolver.  Keeping the
// formulas identical makes A* versus IDA* a search-order comparison rather
// than a heuristic-quality comparison.
int lower_bound2(const State& state, const bool with_pairs) {
    std::array<int, kMaxStacks> count{};
    for (int s = 0; s < state.count; ++s)
        for (int p = 0; p < state.size[s]; ++p) ++count[state.cells[s][p]];

    std::array<int, kMaxStacks> home_run{};
    std::array<int, kMaxStacks> prefix{};
    for (int s = 0; s < state.count; ++s) {
        int p = 0;
        while (p < state.size[s] && state.cells[s][p] == s) ++p;
        home_run[s] = p;
        const bool overflow = (state.overflow_mask & (1U << s)) != 0;
        if (!overflow && p == count[s]) {
            while (p < state.size[s]
                   && (state.overflow_mask & (1U << state.cells[s][p])) != 0) ++p;
        }
        prefix[s] = p;
    }

    int bad = 0;
    std::array<int, kMaxStacks> out_bad{};
    for (int s = 0; s < state.count; ++s) {
        bad += state.size[s] - prefix[s];
        for (int p = prefix[s]; p < state.size[s]; ++p) {
            const int type = state.cells[s][p];
            if (type != s) ++out_bad[type];
        }
    }

    int extra = 0;
    for (int t = 0; t < state.count; ++t) {
        const bool overflow = (state.overflow_mask & (1U << t)) != 0;
        const int need = (overflow ? state.capacity[t] : count[t]) - home_run[t];
        extra += std::max(0, need - out_bad[t]);
    }

    int pair_extra = 0;
    if (with_pairs) {
        std::array<std::array<int, kMaxStacks>, kMaxStacks> cross{};
        for (int s = 0; s < state.count; ++s)
            for (int p = 0; p < state.size[s]; ++p) {
                const int type = state.cells[s][p];
                if (type != s) ++cross[s][type];
            }
        for (int u = 0; u < state.count; ++u) {
            if ((state.overflow_mask & (1U << u)) != 0) continue;
            for (int t = u + 1; t < state.count; ++t) {
                if ((state.overflow_mask & (1U << t)) != 0) continue;
                pair_extra += std::min(cross[u][t], cross[t][u]);
            }
        }
    }
    return (bad + extra + pair_extra) * 2;
}

int estimate_h2(const State& state, const IdaLbMode mode) {
    if (mode == IdaLbMode::kLegacy) return legacy_h2(state);
    return lower_bound2(state, mode == IdaLbMode::kTt);
}

void replace_hash(State& state, const int stack, const int position,
                  const int old_value, const int new_value) {
    state.hash.first ^= kZobrist.first[stack][position][old_value]
        ^ kZobrist.first[stack][position][new_value];
    state.hash.second ^= kZobrist.second[stack][position][old_value]
        ^ kZobrist.second[stack][position][new_value];
}

void apply(State& state, const int src, const int dst) {
    const int src_position = state.size[src] - 1;
    const int dst_position = state.size[dst];
    const int item = state.cells[src][src_position];
    replace_hash(state, src, src_position, item, kEmpty);
    replace_hash(state, dst, dst_position, kEmpty, item);
    --state.size[src];
    state.cells[dst][state.size[dst]++] = static_cast<std::uint8_t>(item);
}

std::optional<State> make_state(const StackProblem& problem, const int max_capacity) {
    if (problem.stacks.empty() || problem.stacks.size() > kMaxStacks
        || problem.stacks.size() != problem.capacities.size()) return std::nullopt;
    const int accept = std::min(max_capacity, kMaxCapacity);
    State state;
    state.count = static_cast<int>(problem.stacks.size());
    std::array<int, kMaxStacks> item_counts{};
    for (int s = 0; s < state.count; ++s) {
        if (problem.capacities[s] < 0 || problem.capacities[s] > accept
            || problem.stacks[s].size() > static_cast<std::size_t>(problem.capacities[s]))
            return std::nullopt;
        state.capacity[s] = static_cast<std::uint8_t>(problem.capacities[s]);
        state.size[s] = static_cast<std::uint8_t>(problem.stacks[s].size());
        for (int p = 0; p < problem.capacities[s]; ++p) {
            const int value = p < static_cast<int>(problem.stacks[s].size())
                ? problem.stacks[s][p] : kEmpty;
            if (value < 0 || (value >= state.count && value != kEmpty)) return std::nullopt;
            state.cells[s][p] = static_cast<std::uint8_t>(value);
            state.hash.first ^= kZobrist.first[s][p][value];
            state.hash.second ^= kZobrist.second[s][p][value];
            if (value != kEmpty) ++item_counts[value];
        }
    }
    for (int t = 0; t < state.count; ++t)
        if (item_counts[t] > problem.capacities[t]) state.overflow_mask |= 1U << t;
    return state;
}

double priority(const BestFirstOptions& options, const int depth, const int h2) {
    const double g2 = static_cast<double>(depth * 2);
    switch (options.mode) {
    case BestFirstMode::kAStar: return g2 + h2;
    case BestFirstMode::kWeightedAStar: return g2 + options.heuristic_weight * h2;
    case BestFirstMode::kGreedyBestFirst: return h2;
    case BestFirstMode::kUniformCost: return g2;
    }
    return g2 + h2;
}

std::vector<StackMove> reconstruct(const std::vector<Node>& nodes, std::size_t index) {
    std::vector<StackMove> path;
    while (nodes[index].parent != std::numeric_limits<std::size_t>::max()) {
        path.push_back(nodes[index].move);
        index = nodes[index].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace

std::string_view BestFirstSolver::name() const noexcept {
    switch (options_.mode) {
    case BestFirstMode::kAStar: return "astar";
    case BestFirstMode::kWeightedAStar: return "wastar";
    case BestFirstMode::kGreedyBestFirst: return "gbfs";
    case BestFirstMode::kUniformCost: return "ucs";
    }
    return "best_first";
}

std::optional<std::vector<StackMove>> BestFirstSolver::solve(
    const StackProblem& problem, SolverStats* const stats) {
    SolverStats local;
    SolverStats& out = stats ? *stats : local;
    const auto started = std::chrono::steady_clock::now();
    const auto finish = [&](const std::string_view outcome) {
        out.outcome = outcome;
        out.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    };

    if (options_.max_expanded_nodes == 0 || options_.max_capacity < 0
        || !std::isfinite(options_.heuristic_weight) || options_.heuristic_weight <= 0.0) {
        finish("rejected");
        return std::nullopt;
    }
    auto initial = make_state(problem, options_.max_capacity);
    if (!initial) {
        finish("rejected");
        return std::nullopt;
    }
    if (solved(*initial)) {
        finish("solved");
        return std::vector<StackMove>{};
    }

    std::vector<Node> nodes;
    nodes.reserve(std::min<std::size_t>(options_.max_expanded_nodes, 4096));
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, Worse> frontier;
    std::unordered_map<Key, int, KeyHash> best_depth;
    best_depth.max_load_factor(0.8F);
    best_depth.reserve(std::min<std::size_t>(options_.max_expanded_nodes, 4096));

    nodes.push_back({*initial});
    const int initial_h2 = estimate_h2(*initial, options_.lb_mode);
    frontier.push({priority(options_, 0, initial_h2), initial_h2, 0, 0});
    best_depth.emplace(initial->hash, 0);

    while (!frontier.empty()) {
        const QueueEntry entry = frontier.top();
        frontier.pop();
        const Node current = nodes[entry.node];
        const auto best = best_depth.find(current.state.hash);
        if (best == best_depth.end() || best->second != current.depth) continue;

        if (++out.expanded_nodes > options_.max_expanded_nodes) {
            finish("iteration_limit");
            return std::nullopt;
        }
        out.iterations = std::max(out.iterations, static_cast<std::uint64_t>(current.depth));
        if (solved(current.state)) {
            auto path = reconstruct(nodes, entry.node);
            out.solution_length = path.size();
            finish("solved");
            return path;
        }

        for (int src = 0; src < current.state.count; ++src) {
            if (current.state.size[src] == 0) continue;
            for (int dst = 0; dst < current.state.count; ++dst) {
                if (src == dst || current.state.size[dst] >= current.state.capacity[dst]) continue;
                if (current.move.first == dst && current.move.second == src) continue;

                State next = current.state;
                apply(next, src, dst);
                const int next_depth = current.depth + 1;
                if (const auto seen = best_depth.find(next.hash);
                    seen != best_depth.end() && seen->second <= next_depth) continue;
                // Best-first methods retain their frontier.  Bound retained
                // nodes as well as expansions so a dense Gate A cell cannot
                // exhaust host memory before the watchdog can intervene.
                if (nodes.size() >= options_.max_expanded_nodes) {
                    finish("iteration_limit");
                    return std::nullopt;
                }
                best_depth[next.hash] = next_depth;
                const std::size_t index = nodes.size();
                nodes.push_back({std::move(next), entry.node, {src, dst}, next_depth});
                const int h2 = estimate_h2(nodes.back().state, options_.lb_mode);
                frontier.push({priority(options_, next_depth, h2), h2, next_depth, index});
            }
        }
    }

    finish("no_solution");
    return std::nullopt;
}

}  // namespace lima
