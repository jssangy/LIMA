#pragma once

#include "lima/core/agent.hpp"
#include "lima/core/grid_map.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

namespace lima {

// Computes one collision-free PIBT move for each eligible agent. Agents that
// are not eligible are treated as fixed obstacles for this timestep.
class PibtResolver {
public:
    // `relaxed` is false on the normal candidate pass and true on the optional
    // second pass, which an agent takes only when every strict candidate
    // failed, i.e. when it would otherwise be forced to stay in place.
    // Callers without a relaxed rule ignore the argument, which keeps the
    // second pass a no-op and the resulting assignment unchanged.
    using CandidateAllowed = std::function<bool(AgentId, CellId, bool)>;

    // age_rate: when true, waiting agents accrue priority age at distinct
    // per-agent rates (1 + rank mod 8) instead of a uniform +1, so a frozen
    // equal-age relative order eventually rotates (straggler-fix variant).
    // relaxed_pass: enable the second (last-resort) candidate pass.  Left off,
    // assign() behaves exactly as the single-pass original, including its
    // consumption of the shared RNG stream.
    PibtResolver(const GridMap& map, std::uint64_t seed, bool age_rate = false,
                 bool relaxed_pass = false);

    void resolve(std::span<const Agent> agents,
                 std::span<const AgentId> occupancy,
                 std::span<const std::uint8_t> eligible,
                 std::span<const std::uint8_t> priority_class,
                 const CandidateAllowed& candidate_allowed,
                 std::vector<CellId>& next_positions);

private:
    const GridMap& map_;
    std::mt19937_64 rng_;
    bool age_rate_{false};
    bool relaxed_pass_{false};
    std::vector<std::uint64_t> priority_age_;
    std::vector<std::uint32_t> initial_distance_;
    std::vector<std::uint32_t> priority_rank_;
    std::vector<CellId> next_positions_;
    std::vector<AgentId> reserved_by_;
    std::vector<AgentId> order_;

    std::span<const Agent> agents_;
    std::span<const AgentId> occupancy_;
    std::span<const std::uint8_t> eligible_;
    std::span<const std::uint8_t> priority_class_;
    const CandidateAllowed* candidate_allowed_{};

    void resize(std::size_t agent_count);
    [[nodiscard]] bool assign(AgentId agent);
    [[nodiscard]] bool try_candidates(AgentId agent, bool relaxed);
    [[nodiscard]] int candidate_distance(const Agent& agent, CellId candidate) const noexcept;
};

}  // namespace lima
