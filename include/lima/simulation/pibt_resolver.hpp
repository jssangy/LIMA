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
    using CandidateAllowed = std::function<bool(AgentId, CellId)>;

    PibtResolver(const GridMap& map, std::uint64_t seed);

    void resolve(std::span<const Agent> agents,
                 std::span<const AgentId> occupancy,
                 std::span<const std::uint8_t> eligible,
                 std::span<const std::uint8_t> priority_class,
                 const CandidateAllowed& candidate_allowed,
                 std::vector<CellId>& next_positions);

private:
    const GridMap& map_;
    std::mt19937_64 rng_;
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
    [[nodiscard]] int candidate_distance(const Agent& agent, CellId candidate) const noexcept;
};

}  // namespace lima
