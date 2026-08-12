#include "lima/scheduling/beam_search.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lima {
namespace {

constexpr int kMaxStacks = 4;
constexpr int kMaxCapacity = 16;

struct Key {
    std::uint64_t first{};
    std::uint64_t second{};
    friend bool operator==(const Key&, const Key&) = default;
    friend bool operator<(const Key& lhs, const Key& rhs) {
        return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
    }
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
    std::array<std::array<std::array<std::uint64_t, kMaxStacks + 1>, kMaxCapacity>, kMaxStacks> first{};
    std::array<std::array<std::array<std::uint64_t, kMaxStacks + 1>, kMaxCapacity>, kMaxStacks> second{};

    Zobrist() {
        std::uint64_t a = 0xa243f6a8885a308dULL;
        std::uint64_t b = 0x13198a2e03707344ULL;
        for (int stack = 0; stack < kMaxStacks; ++stack)
            for (int position = 0; position < kMaxCapacity; ++position)
                for (int value = 0; value <= kMaxStacks; ++value) {
                    first[stack][position][value] = splitmix64(a);
                    second[stack][position][value] = splitmix64(b);
                }
    }
};

const Zobrist kZobrist;

struct State {
    std::uint8_t count{};
    std::array<std::uint8_t, kMaxStacks> capacity{};
    std::array<std::uint8_t, kMaxStacks> size{};
    std::array<std::array<std::uint8_t, kMaxCapacity>, kMaxStacks> cells{};
    std::uint8_t overflow_mask{};
    Key hash{};
};

struct PathNode {
    std::size_t parent{std::numeric_limits<std::size_t>::max()};
    StackMove move{-1, -1};
};

struct FrontierNode {
    State state;
    std::size_t path{};
    StackMove move{-1, -1};
};

struct Candidate {
    State state;
    std::size_t parent{};
    StackMove move;
    int score{};
    std::uint8_t move_rank{};
};

struct Workspace {
    std::pmr::unsynchronized_pool_resource expanded_resource;
    std::pmr::unsynchronized_pool_resource candidate_resource;
    std::pmr::unordered_map<Key, std::size_t, KeyHash> expanded_depth{&expanded_resource};
    std::pmr::unordered_map<Key, std::size_t, KeyHash> candidate_index{&candidate_resource};
    std::vector<PathNode> paths;
    std::vector<FrontierNode> frontier;
    std::vector<Candidate> candidates;

    Workspace() {
        expanded_depth.max_load_factor(0.8F);
        candidate_index.max_load_factor(0.8F);
    }
};

Workspace& workspace() {
    thread_local Workspace value;
    return value;
}

template <class Map>
void ensure_capacity(Map& map, const std::size_t entries) {
    const auto available = static_cast<std::size_t>(
        static_cast<double>(map.bucket_count()) * map.max_load_factor());
    if (available < entries) map.reserve(entries);
}

bool solved(const State& state) {
    if (state.overflow_mask == 0) {
        for (int stack = 0; stack < state.count; ++stack)
            for (int position = 0; position < state.size[stack]; ++position)
                if (state.cells[stack][position] != stack) return false;
        return true;
    }
    for (int stack = 0; stack < state.count; ++stack) {
        const bool overflow = (state.overflow_mask & (1U << stack)) != 0;
        if (overflow) {
            if (state.size[stack] != state.capacity[stack]) return false;
            for (int position = 0; position < state.size[stack]; ++position)
                if (state.cells[stack][position] != stack) return false;
        } else {
            int position = 0;
            while (position < state.size[stack] && state.cells[stack][position] == stack) ++position;
            while (position < state.size[stack]
                   && (state.overflow_mask & (1U << state.cells[stack][position])) != 0) ++position;
            if (position != state.size[stack]) return false;
        }
    }
    return true;
}

bool target_ready(const State& state, const int target) {
    for (int position = 0; position < state.size[target]; ++position)
        if (state.cells[target][position] != target) return false;
    return true;
}

int disorder_score(const State& state) {
    int score = 0;
    for (int stack = 0; stack < state.count; ++stack) {
        if ((state.overflow_mask & (1U << stack)) != 0) {
            score += (state.capacity[stack] - state.size[stack]) * 8;
            for (int position = 0; position < state.size[stack]; ++position) {
                if (state.cells[stack][position] != stack)
                    score += 12 + (state.size[stack] - position - 1) * 2;
            }
            continue;
        }

        bool suffix = false;
        for (int position = 0; position < state.size[stack]; ++position) {
            const int item = state.cells[stack][position];
            const bool acceptable_overflow = (state.overflow_mask & (1U << item)) != 0;
            if (!suffix && item == stack) continue;
            suffix = true;
            if (!acceptable_overflow)
                score += 10 + (state.size[stack] - position - 1) * 2;
            else score += 1;
        }
    }
    return score;
}

void replace_hash(State& state, const int stack, const int position,
                  const int old_value, const int new_value) {
    state.hash.first ^= kZobrist.first[stack][position][old_value]
        ^ kZobrist.first[stack][position][new_value];
    state.hash.second ^= kZobrist.second[stack][position][old_value]
        ^ kZobrist.second[stack][position][new_value];
}

void apply(State& state, const int src, const int dst) {
    const int empty = state.count;
    const int src_position = state.size[src] - 1;
    const int dst_position = state.size[dst];
    const int item = state.cells[src][src_position];
    replace_hash(state, src, src_position, item, empty);
    replace_hash(state, dst, dst_position, empty, item);
    --state.size[src];
    state.cells[dst][state.size[dst]++] = static_cast<std::uint8_t>(item);
}

std::uint8_t move_rank(const State& state, const int src, const int dst) {
    const int item = state.cells[src][state.size[src] - 1];
    if (dst == item && target_ready(state, dst)) return 0;
    if (state.size[dst] > 0 && state.cells[dst][state.size[dst] - 1] == item) return 1;
    if (state.size[dst] == 0) return 2;
    return 3;
}

State make_state(const std::vector<std::vector<int>>& stacks,
                 const std::vector<int>& capacities) {
    if (stacks.empty() || stacks.size() > kMaxStacks || stacks.size() != capacities.size())
        throw std::invalid_argument("stack count must be 1..4 and match capacities");

    State state;
    state.count = static_cast<std::uint8_t>(stacks.size());
    std::array<int, kMaxStacks> item_counts{};
    for (int stack = 0; stack < state.count; ++stack) {
        if (capacities[stack] < 0 || capacities[stack] > kMaxCapacity
            || stacks[stack].size() > static_cast<std::size_t>(capacities[stack]))
            throw std::invalid_argument("invalid stack capacity");
        state.capacity[stack] = static_cast<std::uint8_t>(capacities[stack]);
        state.size[stack] = static_cast<std::uint8_t>(stacks[stack].size());
        for (int position = 0; position < capacities[stack]; ++position) {
            const int value = position < static_cast<int>(stacks[stack].size())
                ? stacks[stack][position] : state.count;
            if (value < 0 || (position < static_cast<int>(stacks[stack].size()) && value >= state.count))
                throw std::invalid_argument("invalid target stack index");
            state.cells[stack][position] = static_cast<std::uint8_t>(value);
            state.hash.first ^= kZobrist.first[stack][position][value];
            state.hash.second ^= kZobrist.second[stack][position][value];
            if (value != state.count) ++item_counts[value];
        }
    }
    for (int item = 0; item < state.count; ++item)
        if (item_counts[item] > capacities[item]) state.overflow_mask |= 1U << item;
    return state;
}

std::vector<StackMove> reconstruct(const std::vector<PathNode>& paths, std::size_t index) {
    std::vector<StackMove> path;
    while (paths[index].parent != std::numeric_limits<std::size_t>::max()) {
        path.push_back(paths[index].move);
        index = paths[index].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace

std::vector<StackMove> solve_stack_rearrangement_beam(
    const std::vector<std::vector<int>>& stacks, const std::vector<int>& capacities,
    const BeamSearchOptions options) {
    if (options.beam_width == 0 || options.max_depth == 0 || options.max_expanded_nodes == 0)
        throw std::invalid_argument("beam search limits must be positive");

    State initial = make_state(stacks, capacities);
    if (solved(initial)) return {};

    Workspace& work = workspace();
    auto& paths = work.paths;
    auto& frontier = work.frontier;
    auto& candidates = work.candidates;
    auto& expanded_depth = work.expanded_depth;
    auto& candidate_index = work.candidate_index;

    paths.clear();
    frontier.clear();
    candidates.clear();
    expanded_depth.clear();
    candidate_index.clear();

    // Most intersection schedules are shallow. Start with a modest retained
    // workspace and let it grow only when a genuinely hard instance needs it;
    // the previous implementation allocated for 131k path states and one
    // million hash buckets on every scheduler invocation.
    const std::size_t warm_entries = std::min(
        options.max_expanded_nodes, options.beam_width * std::size_t{8} + 1);
    if (paths.capacity() < warm_entries) paths.reserve(warm_entries);
    if (frontier.capacity() < options.beam_width) frontier.reserve(options.beam_width);
    ensure_capacity(expanded_depth, warm_entries);

    paths.push_back({std::numeric_limits<std::size_t>::max(), {-1, -1}});
    frontier.push_back({initial, 0, {-1, -1}});
    expanded_depth.emplace(initial.hash, 0);
    std::size_t expanded = 0;

    std::size_t searched_depth = 0;
    for (; searched_depth < options.max_depth && !frontier.empty(); ++searched_depth) {
        const std::size_t candidate_capacity =
            frontier.size() * kMaxStacks * (kMaxStacks - 1);
        candidates.clear();
        if (candidates.capacity() < candidate_capacity) candidates.reserve(candidate_capacity);
        candidate_index.clear();
        ensure_capacity(candidate_index, candidate_capacity);

        for (const FrontierNode& node : frontier) {
            if (++expanded > options.max_expanded_nodes)
                throw std::runtime_error("Beam search: expanded-node limit exceeded");
            for (int src = 0; src < node.state.count; ++src) {
                if (node.state.size[src] == 0) continue;
                for (int dst = 0; dst < node.state.count; ++dst) {
                    if (src == dst || node.state.size[dst] >= node.state.capacity[dst]) continue;
                    if (node.move.first == dst && node.move.second == src) continue;

                    const std::uint8_t rank = move_rank(node.state, src, dst);
                    State next = node.state;
                    apply(next, src, dst);
                    if (solved(next)) {
                        paths.push_back({node.path, {src, dst}});
                        return reconstruct(paths, paths.size() - 1);
                    }
                    if (const auto seen = expanded_depth.find(next.hash);
                        seen != expanded_depth.end() && seen->second <= searched_depth + 1) continue;

                    Candidate candidate{
                        std::move(next), node.path, {src, dst}, 0, rank};
                    candidate.score = disorder_score(candidate.state);
                    const auto [found, inserted] = candidate_index.emplace(
                        candidate.state.hash, candidates.size());
                    if (inserted) candidates.push_back(std::move(candidate));
                    else {
                        Candidate& previous = candidates[found->second];
                        if (candidate.score < previous.score
                            || (candidate.score == previous.score && candidate.move_rank < previous.move_rank))
                            previous = std::move(candidate);
                    }
                }
            }
        }

        const auto better = [](const Candidate& lhs, const Candidate& rhs) {
            if (lhs.score != rhs.score) return lhs.score < rhs.score;
            if (lhs.move_rank != rhs.move_rank) return lhs.move_rank < rhs.move_rank;
            if (lhs.state.hash != rhs.state.hash) return lhs.state.hash < rhs.state.hash;
            return lhs.move < rhs.move;
        };
        if (candidates.size() > options.beam_width) {
            std::nth_element(candidates.begin(),
                candidates.begin() + static_cast<std::ptrdiff_t>(options.beam_width),
                candidates.end(), better);
            candidates.resize(options.beam_width);
        }
        std::sort(candidates.begin(), candidates.end(), better);

        frontier.clear();
        frontier.reserve(candidates.size());
        for (Candidate& candidate : candidates) {
            expanded_depth[candidate.state.hash] = searched_depth + 1;
            paths.push_back({candidate.parent, candidate.move});
            frontier.push_back({std::move(candidate.state), paths.size() - 1, candidate.move});
        }
    }
    throw std::runtime_error("Beam search: no solution within search limits (depth="
        + std::to_string(searched_depth) + ", expanded=" + std::to_string(expanded) + ")");
}

}  // namespace lima
