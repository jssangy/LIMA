#include "lima/scheduling/ida_star.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <unordered_map>

namespace lima {
namespace {

constexpr int kMaxStacks = 4;
constexpr int kMaxCapacity = 64;  // raised from 16 for long-corridor stress studies
constexpr int kEmpty = 16;        // sentinel value; item values stay in 0..3

struct Key {
    std::uint64_t first{};
    std::uint64_t second{};
    friend bool operator==(const Key&, const Key&) = default;
};

struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
        std::uint64_t value = key.first ^ (key.second + 0x9e3779b97f4a7c15ULL + (key.first << 6U) + (key.first >> 2U));
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
        std::uint64_t a = 0x12345678abcdef01ULL;
        std::uint64_t b = 0x0fedcba987654321ULL;
        for (int s = 0; s < kMaxStacks; ++s) for (int p = 0; p < kMaxCapacity; ++p) for (int v = 0; v <= 16; ++v) {
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

void replace_hash(State& state, const int stack, const int position, const int old_value, const int new_value) {
    state.hash.first ^= kZobrist.first[stack][position][old_value] ^ kZobrist.first[stack][position][new_value];
    state.hash.second ^= kZobrist.second[stack][position][old_value] ^ kZobrist.second[stack][position][new_value];
}

bool target_ready(const State& state, const int target) {
    for (int i = 0; i < state.size[target]; ++i) if (state.cells[target][i] != target) return false;
    return true;
}

bool solved(const State& state) {
    if (state.overflow_mask == 0) {
        for (int s = 0; s < state.count; ++s) for (int p = 0; p < state.size[s]; ++p) {
            if (state.cells[s][p] != s) return false;
        }
        return true;
    }
    for (int s = 0; s < state.count; ++s) {
        const bool overflow = (state.overflow_mask & (1U << s)) != 0;
        if (overflow) {
            if (state.size[s] != state.capacity[s]) return false;
            for (int p = 0; p < state.size[s]; ++p) if (state.cells[s][p] != s) return false;
        } else {
            int p = 0;
            while (p < state.size[s] && state.cells[s][p] == s) ++p;
            while (p < state.size[s] && (state.overflow_mask & (1U << state.cells[s][p])) != 0) ++p;
            if (p != state.size[s]) return false;
        }
    }
    return true;
}

int heuristic2(const State& state) {
    int value = 0;
    for (int s = 0; s < state.count; ++s) {
        bool clean = true;
        for (int depth = 0; depth < state.size[s]; ++depth) {
            if (clean && state.cells[s][depth] != s) {
                clean = false;
                value += (state.capacity[s] - depth) * 3;
            } else if (!clean) {
                value += 2;
            }
        }
    }
    return value;
}

struct Undo { std::uint8_t item{}, src{}, dst{}, src_position{}, dst_position{}; };

void apply(State& state, const int src, const int dst, Undo& undo) {
    undo = {state.cells[src][state.size[src] - 1], static_cast<std::uint8_t>(src), static_cast<std::uint8_t>(dst),
            static_cast<std::uint8_t>(state.size[src] - 1), state.size[dst]};
    replace_hash(state, src, undo.src_position, undo.item, kEmpty);
    replace_hash(state, dst, undo.dst_position, kEmpty, undo.item);
    --state.size[src];
    state.cells[dst][state.size[dst]++] = undo.item;
}

void rollback(State& state, const Undo& undo) {
    --state.size[undo.dst];
    ++state.size[undo.src];
    replace_hash(state, undo.dst, undo.dst_position, undo.item, kEmpty);
    replace_hash(state, undo.src, undo.src_position, kEmpty, undo.item);
    state.cells[undo.src][undo.src_position] = undo.item;
}

struct MoveList {
    std::array<StackMove, kMaxStacks * (kMaxStacks - 1)> values{};
    std::size_t size{};

    void push_back(const StackMove move) { values[size++] = move; }
};

MoveList generate_moves(const State& state, const StackMove last, const bool deterministic) {
    MoveList result;
    if (deterministic) {
        for (int src = 0; src < state.count; ++src) {
            if (state.size[src] == 0) continue;
            const int dst = state.cells[src][state.size[src] - 1];
            if (dst != src && dst < state.count && state.size[dst] < state.capacity[dst] && target_ready(state, dst)
                && !(src == last.second && dst == last.first)) {
                result.push_back({src, dst});
                return result;
            }
        }
    }
    std::array<StackMove, kMaxStacks * (kMaxStacks - 1)> unordered_moves{};
    std::array<std::uint8_t, kMaxStacks * (kMaxStacks - 1)> buckets{};
    std::size_t count = 0;
    for (int src = 0; src < state.count; ++src) {
        if (state.size[src] == 0) continue;
        const int item = state.cells[src][state.size[src] - 1];
        for (int dst = 0; dst < state.count; ++dst) {
            if (src == dst || state.size[dst] >= state.capacity[dst] || (src == last.second && dst == last.first)) continue;
            int bucket = 3;
            if (dst == item && target_ready(state, dst)) bucket = 0;
            else if (state.size[dst] > 0 && state.cells[dst][state.size[dst] - 1] == item) bucket = 1;
            else if (state.size[dst] == 0) bucket = 2;
            unordered_moves[count] = {src, dst};
            buckets[count++] = static_cast<std::uint8_t>(bucket);
        }
    }
    for (std::uint8_t bucket = 0; bucket < 4; ++bucket) {
        for (std::size_t i = 0; i < count; ++i) {
            if (buckets[i] == bucket) result.push_back(unordered_moves[i]);
        }
    }
    return result;
}

struct SearchResult { int bound2; bool found; };

using Visited = std::pmr::unordered_map<Key, int, KeyHash>;

SearchResult search(State& state, const int depth, const int bound2,
                    Visited& visited, std::vector<StackMove>& path,
                    const bool deterministic, std::uint64_t& expanded) {
    ++expanded;
    if (const auto found = visited.find(state.hash); found != visited.end() && found->second <= depth) {
        return {std::numeric_limits<int>::max(), false};
    }
    visited[state.hash] = depth;
    const int estimate2 = depth * 2 + heuristic2(state);
    if (estimate2 > bound2) return {estimate2, false};
    if (solved(state)) return {estimate2, true};

    const StackMove last = path.empty() ? StackMove{-1, -1} : path.back();
    const MoveList moves = generate_moves(state, last, deterministic);
    int next_bound = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < moves.size; ++i) {
        const StackMove move = moves.values[i];
        Undo undo;
        apply(state, move.first, move.second, undo);
        path.push_back(move);
        const SearchResult result = search(state, depth + 1, bound2, visited, path, deterministic, expanded);
        if (result.found) return result;
        next_bound = std::min(next_bound, result.bound2);
        path.pop_back();
        rollback(state, undo);
    }
    return {next_bound, false};
}

// Returns nullopt when the instance violates the representation limits.
std::optional<State> make_state(const std::vector<std::vector<int>>& stacks, const std::vector<int>& capacities) {
    if (stacks.empty() || stacks.size() > kMaxStacks || stacks.size() != capacities.size()) {
        return std::nullopt;
    }
    State state;
    state.count = static_cast<int>(stacks.size());
    std::array<int, kMaxStacks> item_counts{};
    for (int s = 0; s < state.count; ++s) {
        if (capacities[s] < 0 || capacities[s] > kMaxCapacity || stacks[s].size() > static_cast<std::size_t>(capacities[s])) {
            return std::nullopt;
        }
        state.capacity[s] = static_cast<std::uint8_t>(capacities[s]);
        state.size[s] = static_cast<std::uint8_t>(stacks[s].size());
        for (int p = 0; p < capacities[s]; ++p) {
            const int value = p < static_cast<int>(stacks[s].size()) ? stacks[s][p] : kEmpty;
            if (value < 0 || (value >= state.count && value != kEmpty)) return std::nullopt;
            state.cells[s][p] = static_cast<std::uint8_t>(value);
            state.hash.first ^= kZobrist.first[s][p][value];
            state.hash.second ^= kZobrist.second[s][p][value];
            if (value != kEmpty) ++item_counts[value];
        }
    }
    for (int type = 0; type < state.count; ++type) if (item_counts[type] > capacities[type]) state.overflow_mask |= 1U << type;
    return state;
}

}  // namespace

std::optional<std::vector<StackMove>> IdaStarSolver::solve(
    const StackProblem& problem, SolverStats* const stats) {
    SolverStats local;
    SolverStats& out = stats ? *stats : local;
    const auto started = std::chrono::steady_clock::now();
    const auto finish = [&](const std::string_view outcome) {
        out.outcome = outcome;
        out.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };

    auto state_opt = make_state(problem.stacks, problem.capacities);
    if (!state_opt) {
        finish("rejected");
        return std::nullopt;
    }
    State state = *state_opt;
    const int initial_bound = heuristic2(state);
    int bound = initial_bound;
    bool deterministic = options_.deterministic_move;
    std::vector<StackMove> path;
    path.reserve(256);
    std::pmr::unsynchronized_pool_resource visited_pool;
    Visited visited{&visited_pool};
    visited.reserve(200'000);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        ++out.iterations;
        visited.clear();
        const SearchResult result = search(state, 0, bound, visited, path, deterministic, out.expanded_nodes);
        if (result.found) {
            out.solution_length = path.size();
            out.fastpath_solved = deterministic;
            finish("solved");
            return path;
        }
        if (result.bound2 == std::numeric_limits<int>::max()) {
            if (deterministic) {
                deterministic = false;
                bound = initial_bound;
                path.clear();
                iteration = static_cast<std::size_t>(-1);
                continue;
            }
            finish("no_solution");
            return std::nullopt;
        }
        bound = options_.bound_step > 0 ? std::max(result.bound2, bound + options_.bound_step) : result.bound2;
        path.clear();
    }
    finish("iteration_limit");
    return std::nullopt;
}

std::vector<StackMove> solve_stack_rearrangement(const std::vector<std::vector<int>>& stacks,
                                                 const std::vector<int>& capacities,
                                                 const IdaStarOptions options) {
    IdaStarSolver solver(options);
    SolverStats stats;
    auto result = solver.solve({stacks, capacities}, &stats);
    if (!result) throw std::runtime_error("IDA*: " + std::string(stats.outcome));
    return std::move(*result);
}

}  // namespace lima
