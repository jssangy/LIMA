#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"
#include "lima/intersection/coordinator.hpp"
#include "lima/planning/planner.hpp"
#include "lima/simulation/goal_allocator.hpp"
#include "lima/simulation/pibt_resolver.hpp"

#include <memory>
#include <random>
#include <array>
#include <cstdint>
#include <span>
#include <unordered_set>

namespace lima {

class DebugTrace;

enum class WorkloadMode : std::uint8_t { OneShot, Lifelong };

struct SimulationStats {
    std::uint64_t timestep{};
    std::uint64_t completed{};
    std::uint64_t completed_tasks{};
    std::uint64_t total_task_latency{};
    std::uint64_t committed_moves{};
    std::uint64_t waits{};
    std::uint64_t detected_deadlocks{};
};

enum class ScheduleDecision : std::uint8_t {
    NotChecked,
    Active,
    NoDeadlock,
    NeighborActive,
    Candidate,
    CapacityExceeded,
    NeighborQuotaBlocked,
    PlannerFailed,
    Activated,
};

class Simulator {
public:
    Simulator(GridMap map, std::span<const Task> tasks, PlannerKind planner_kind,
              std::uint64_t seed = 0, WorkloadMode workload = WorkloadMode::OneShot);

    [[nodiscard]] bool step();
    [[nodiscard]] bool done() const noexcept {
        return workload_ == WorkloadMode::OneShot && stats_.completed == agents_.size();
    }
    [[nodiscard]] bool lifelong() const noexcept { return workload_ == WorkloadMode::Lifelong; }
    [[nodiscard]] const SimulationStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const GridMap& map() const noexcept { return map_; }
    [[nodiscard]] const std::vector<Agent>& agents() const noexcept { return agents_; }
    [[nodiscard]] const IntersectionTopology& topology() const noexcept { return topology_; }

private:
    friend class DebugTrace;

    struct PendingSchedule {
        IntersectionId intersection{-1};
        std::vector<ScheduledPath> paths;
    };

    GridMap map_;
    std::mt19937_64 rng_;
    std::unique_ptr<Planner> planner_;
    IntersectionTopology topology_;
    std::vector<Agent> agents_;
    WorkloadMode workload_{WorkloadMode::OneShot};
    bool despawn_at_goal_{};
    std::unique_ptr<GoalAllocator> goal_allocator_;
    IntersectionCoordinator coordinator_;
    std::vector<IntersectionId> deadlock_queue_;
    std::vector<std::unordered_set<AgentId>> scheduled_members_;
    std::vector<bool> deadlock_waiting_;
    std::vector<int> intersection_available_;
    std::vector<int> intersection_capacity_;
    std::vector<std::uint8_t> deadlock_active_;
    std::vector<std::uint8_t> deadlock_release_grace_;
    std::vector<std::size_t> deadlock_priority_;

    // Per-timestep workspaces retain their capacity between calls to step().
    std::vector<std::size_t> inside_counts_;
    std::vector<std::vector<AgentId>> members_;
    std::vector<std::vector<IntersectionIntent>> intents_;
    std::vector<std::uint8_t> intent_valid_;
    std::vector<ScheduleDecision> schedule_decision_;
    std::vector<std::array<int, 4>> debug_initial_counts_;
    std::vector<std::array<int, 4>> debug_quotas_;
    std::vector<std::array<int, 4>> debug_final_counts_;
    std::vector<std::uint8_t> debug_quota_valid_;
    std::vector<std::uint8_t> debug_final_valid_;
    std::vector<bool> check_;
    std::vector<bool> stalled_;
    std::vector<bool> blocked_;
    std::vector<AgentId> occupancy_;
    std::vector<bool> normal_occupied_;
    std::vector<std::uint8_t> scheduled_reserved_;
    std::vector<std::uint8_t> rescue_candidate_;
    std::vector<std::uint8_t> rescue_group_;
    std::vector<IntersectionId> rescue_member_;
    std::vector<CellId> movement_origin_;
    std::vector<CellId> movement_intended_;
    std::vector<std::size_t> movement_scheduling_;
    std::vector<std::uint32_t> movement_wait_steps_;
    std::vector<std::uint8_t> pibt_eligible_;
    std::vector<std::uint8_t> pibt_priority_class_;
    std::vector<CellId> pibt_forced_next_;
    std::vector<CellId> pibt_next_;
    std::vector<IntersectionId> candidates_;
    std::vector<PendingSchedule> pending_;
    PibtResolver pibt_;
    SimulationStats stats_;

    bool recover_stalled_intersections(const std::vector<std::vector<AgentId>>& members,
                                       const std::vector<bool>& stalled);
    bool has_active_neighbor(IntersectionId intersection) const;
    void rebuild_deadlock_priorities();
    bool block_intersection(CellId current, CellId next, bool normal_only) const;
    void update_available_on_move(CellId current, CellId next);
    void insert_scheduled_path(Agent& agent, const ScheduledPath& scheduled, IntersectionId intersection);
    void move_agent(Agent& agent);
    void move_agent_to(Agent& agent, CellId next);
    [[nodiscard]] bool adjacent_or_equal(CellId current, CellId next) const;
    [[nodiscard]] CellId active_discharge_target(const Agent& agent) const;
    void assign_lifelong_goals();
    std::vector<CellId> plan_global(CellId start, CellId goal);
};

}  // namespace lima
