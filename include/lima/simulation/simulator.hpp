#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"
#include "lima/intersection/coordinator.hpp"
#include "lima/planning/planner.hpp"
#include "lima/simulation/dependency_resolver.hpp"

#include <memory>
#include <array>
#include <span>
#include <unordered_map>

namespace lima {

struct SimulationStats {
    std::uint64_t timestep{};
    std::uint64_t completed{};
    std::uint64_t committed_moves{};
    std::uint64_t waits{};
    std::uint64_t detected_deadlocks{};
};

class Simulator {
public:
    Simulator(GridMap map, std::span<const Task> tasks, PlannerKind planner_kind);

    [[nodiscard]] bool step();
    [[nodiscard]] bool done() const noexcept { return stats_.completed == agents_.size(); }
    [[nodiscard]] const SimulationStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const GridMap& map() const noexcept { return map_; }
    [[nodiscard]] const std::vector<Agent>& agents() const noexcept { return agents_; }
    [[nodiscard]] const IntersectionTopology& topology() const noexcept { return topology_; }

private:
    GridMap map_;
    std::unique_ptr<Planner> planner_;
    IntersectionTopology topology_;
    std::vector<Agent> agents_;
    DependencyResolver resolver_;
    IntersectionCoordinator coordinator_;
    std::int32_t next_schedule_group_{};
    std::unordered_map<std::int32_t, IntersectionId> active_schedule_intersections_;
    struct DischargeReservation {
        IntersectionId source{-1};
        std::array<int, 4> remaining{};
    };
    std::unordered_map<std::int32_t, DischargeReservation> discharge_reservations_;
    // Persistent admission tokens.  Scheduling reserves downstream tokens before
    // paths are committed; successful boundary crossings return/consume them.
    std::vector<int> intersection_available_;
    std::uint32_t stalled_timesteps_{};
    SimulationStats stats_;

    bool recover_from_stall();
    bool recover_stalled_intersections(const std::vector<std::vector<AgentId>>& members,
                                       const std::vector<bool>& stalled);
    std::vector<CellId> plan_global(CellId start, CellId goal);
    void reconnect_to_original_route(Agent& agent, IntersectionId intersection_id);
};

}  // namespace lima
