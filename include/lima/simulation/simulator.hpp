#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"
#include "lima/intersection/coordinator.hpp"
#include "lima/intersection/gating.hpp"
#include "lima/planning/planner.hpp"
#include "lima/scheduling/solver.hpp"

#include <memory>
#include <random>
#include <array>
#include <cstdint>
#include <span>
#include <unordered_set>

namespace lima {

struct SimulationStats {
    std::uint64_t timestep{};
    std::uint64_t completed{};
    std::uint64_t committed_moves{};
    std::uint64_t waits{};
    std::uint64_t detected_deadlocks{};
};

// Composition of the pluggable pieces.  Defaults reproduce the shipped
// behavior exactly; every knob exists for a specific revision experiment.
struct SimulatorConfig {
    SolverConfig solver{};
    IsolationConfig isolation{};
    bool discharge_enabled{true};
    bool direct_routing{false};  // skip the highway-alignment (DoR-style) global router
};

class Simulator {
public:
    Simulator(GridMap map, std::span<const Task> tasks, PlannerKind planner_kind, std::uint64_t seed = 0,
              SimulatorConfig config = {});

    [[nodiscard]] bool step();
    [[nodiscard]] bool done() const noexcept { return stats_.completed == agents_.size(); }
    [[nodiscard]] const SimulationStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const GridMap& map() const noexcept { return map_; }
    [[nodiscard]] const std::vector<Agent>& agents() const noexcept { return agents_; }
    [[nodiscard]] const IntersectionTopology& topology() const noexcept { return topology_; }
    [[nodiscard]] const SimulatorConfig& config() const noexcept { return config_; }

private:
    struct PendingSchedule {
        IntersectionId intersection{-1};
        std::vector<ScheduledPath> paths;
    };

    GridMap map_;
    SimulatorConfig config_;
    std::mt19937_64 rng_;
    std::unique_ptr<Planner> planner_;
    IntersectionTopology topology_;
    std::vector<Agent> agents_;
    std::unique_ptr<StackSolver> solver_;
    IntersectionCoordinator coordinator_;
    RecirculationDischarge discharge_;
    std::vector<IntersectionId> deadlock_queue_;
    std::vector<std::unordered_set<AgentId>> scheduled_members_;
    std::vector<bool> deadlock_waiting_;
    std::vector<int> intersection_available_;
    std::vector<int> intersection_capacity_;
    std::vector<std::uint8_t> deadlock_active_;
    std::vector<std::size_t> deadlock_priority_;

    // Per-timestep workspaces retain their capacity between calls to step().
    std::vector<std::size_t> inside_counts_;
    std::vector<std::vector<AgentId>> members_;
    std::vector<std::vector<IntersectionIntent>> intents_;
    std::vector<bool> check_;
    std::vector<bool> stalled_;
    std::vector<bool> blocked_;
    std::vector<AgentId> occupancy_;
    std::vector<bool> normal_occupied_;
    std::vector<IntersectionId> candidates_;
    std::vector<PendingSchedule> pending_;
    SimulationStats stats_;

    bool has_active_neighbor(IntersectionId intersection) const;
    void rebuild_deadlock_priorities();
    bool block_intersection(CellId current, CellId next, bool normal_only) const;
    void update_available_on_move(CellId current, CellId next);
    void insert_scheduled_path(Agent& agent, const ScheduledPath& scheduled, IntersectionId intersection);
    void move_agent(Agent& agent);
    std::vector<CellId> plan_global(CellId start, CellId goal);
};

}  // namespace lima
