#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"
#include "lima/intersection/coordinator.hpp"
#include "lima/intersection/gating.hpp"
#include "lima/planning/planner.hpp"
#include "lima/scheduling/solver.hpp"
#include "lima/simulation/goal_allocator.hpp"
#include "lima/simulation/metrics.hpp"
#include "lima/simulation/pibt_resolver.hpp"
#include "lima/simulation/trace.hpp"

#include <memory>
#include <random>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_set>

namespace lima {

struct SimulationStats {
    std::uint64_t timestep{};
    std::uint64_t completed{};
    std::uint64_t committed_moves{};
    std::uint64_t waits{};
    std::uint64_t detected_deadlocks{};
    std::uint64_t command_failures{};
};

enum class GoalBehavior : std::uint8_t {
    Disappear,  // paper default: exit process, agent leaves the system
    Stay,       // agent parks on its goal and becomes an obstacle (E8)
    Lifelong,   // agent immediately receives a fresh goal (E9 task stream)
};

// Composition of the pluggable pieces.  Defaults reproduce the shipped
// behavior exactly; every knob exists for a specific revision experiment.
struct SimulatorConfig {
    SolverConfig solver{};
    IsolationConfig isolation{};
    DischargeConfig discharge{};
    std::uint32_t stall_threshold{10};  // all-active-waiting steps before the run is declared stalled
    bool discharge_enabled{true};
    // Gridlock rotation: when blocked agents form a closed mutual-wait cycle
    // spanning intersections (each one's next cell held by the next agent),
    // advance the whole ring one step simultaneously.  Length >= 3 only, so
    // the no-swap constraint is preserved; the move set is a cyclic
    // permutation, so it is vertex- and edge-conflict free by construction.
    bool rotation_enabled{false};
    // Availability resync: recompute each intersection's admission budget
    // from actual occupancy every step, fixing the credit leak that scheduled
    // exits never repay (M10 root cause candidate).
    // Default since 2026-08-13: admission budgets are recomputed from actual
    // zone occupancy every cycle (stateless, self-healing bookkeeping).
    bool gate_resync{true};
    // Saturated-intersection scheduling: when occupancy exceeds the
    // solvability bound, solve for the innermost bound-many agents per arm and
    // treat deeper cells as walls.  Uses only zone-local occupancy.
    bool subset_scheduling{false};
    // PIBT corridor movement (default since the open-map instance set):
    // agents outside managed intersections move via priority-inheritance
    // backtracking instead of the single-route wait rule.  Also switches the
    // coordinator to its strict release rules (inward-motion guard, tip-only
    // trimmed egress) and enables rescue-group rotation.  Disable with
    // --no-pibt-corridor for ablation.
    bool pibt_corridor{true};
    // Straggler-class fixes (Gate C/D tournament, all opt-in; defaults keep
    // the frozen behavior byte-identical).  Diagnosed mechanism: a sink-cell
    // occupant at an aisle mouth can become unpushable because its only free
    // escape is an intersection-arm cell (forbidden detour), while the
    // sink-seeking root reserves its own cell (PIBT stay) and equal-rate
    // aging freezes the unlucky priority order forever.
    //
    // Sink-yield: an agent whose next intended cell is its own goal, while a
    // different agent (whose goal it is not) stands on that cell, is demoted
    // below every normal agent for this cycle so the occupant's dependency
    // chain can act as root and push its way free.  Uses only the agent's own
    // goal, the occupant of one adjacent cell, and that occupant's goal
    // (zone-local, 1-hop).
    bool pibt_sink_yield{false};
    // Arm-retreat: a PIBT detour candidate inside a managed intersection arm
    // is allowed when that intersection has no active schedule, subject to
    // the same admission gate as route crossings.  Restores original-PIBT
    // pushability at aisle mouths whose only sidestep is an arm cell.
    bool pibt_arm_retreat{false};
    // Last-resort form of the same rule: the arm cell is offered only to an
    // agent that found no strict candidate at all, i.e. one that would
    // otherwise be frozen in place.  Strict traffic is never displaced, so the
    // perturbation stays confined to the deadlocked pair instead of changing
    // routing everywhere (the eager form's failure mode on cross_3030).
    bool pibt_arm_retreat_last{false};
    // Age-rate asymmetry: agents accrue priority age at distinct per-agent
    // rates so frozen equal-age relative orders eventually rotate.
    bool pibt_age_rate{false};
    // Off-route replan: an unscheduled agent displaced from its route
    // (position != route[cursor]) that has waited this many steps recomputes
    // its global route from its current cell (0 = disabled).  Rescues agents
    // parked in a Manhattan local minimum of the waypoint pull.  Must be
    // smaller than stall_threshold to fire before the run is declared stalled.
    std::uint32_t pibt_replan{0};
    // W7 no-prior-priority check: when >= 0, the per-cycle intersection
    // scheduling order (normally least-loaded first, id tie-break) is replaced
    // by a fresh seeded random permutation each cycle.  Completion must be
    // invariant to this order for the Gate B completeness claim to hold.
    std::int64_t shuffle_order{-1};
    // E12 actuator uncertainty: a commanded move is independently dropped
    // with this probability.  A dropped command becomes a wait; coupled move
    // groups and dependency chains are conservatively stopped as needed to
    // preserve vertex/edge safety.
    double failure_probability{0.0};
    GoalBehavior goal_behavior{GoalBehavior::Disappear};
    bool direct_routing{false};   // skip the highway-alignment (DoR-style) global router
    std::string metrics_dir;      // W1 instrumentation output; empty = disabled
    std::string trace_path;       // JSONL step trace for the debug harness; empty = disabled
    std::string map_file;         // recorded in the trace header
};

class Simulator {
public:
    // preset_routes[i], when non-empty, replaces the global router for agent i
    // (external planner injection, e.g. CBS-timeout reference routes); agents
    // without a preset route fall back to the built-in router.
    Simulator(GridMap map, std::span<const Task> tasks, PlannerKind planner_kind, std::uint64_t seed = 0,
              SimulatorConfig config = {},
              std::span<const std::vector<Coord>> preset_routes = {});

    [[nodiscard]] bool step();
    [[nodiscard]] bool done() const noexcept {
        return config_.goal_behavior != GoalBehavior::Lifelong && stats_.completed == agents_.size();
    }
    [[nodiscard]] const SimulationStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const GridMap& map() const noexcept { return map_; }
    [[nodiscard]] const std::vector<Agent>& agents() const noexcept { return agents_; }
    [[nodiscard]] const IntersectionTopology& topology() const noexcept { return topology_; }
    [[nodiscard]] const SimulatorConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool lifelong() const noexcept {
        return config_.goal_behavior == GoalBehavior::Lifelong;
    }

    // Debug-harness views (read only).
    [[nodiscard]] const std::vector<int>& intersection_available() const noexcept { return intersection_available_; }
    [[nodiscard]] const std::vector<int>& intersection_capacity() const noexcept { return intersection_capacity_; }
    [[nodiscard]] const std::vector<std::uint8_t>& deadlock_active() const noexcept { return deadlock_active_; }
    [[nodiscard]] std::vector<bool> deadlock_waiting() const { return deadlock_waiting_; }
    // Returns an empty string when every internal invariant holds, otherwise a
    // description of the first violation.
    [[nodiscard]] std::string check_invariants() const;

    // Writes per-agent metrics; call once after the run when metrics are on.
    void write_metrics();

private:
    struct PendingSchedule {
        IntersectionId intersection{-1};
        std::vector<ScheduledPath> paths;
    };

    GridMap map_;
    SimulatorConfig config_;
    std::mt19937_64 rng_;
    std::mt19937_64 shuffle_rng_;  // W7 only; dedicated so rng_ draws are untouched
    std::mt19937_64 failure_rng_;  // E12 only; dedicated so p=0 preserves every legacy draw
    std::unique_ptr<Planner> planner_;
    IntersectionTopology topology_;
    std::vector<Agent> agents_;
    std::unique_ptr<StackSolver> solver_;
    IntersectionCoordinator coordinator_;
    RecirculationDischarge discharge_;
    std::unique_ptr<MetricsCollector> metrics_;
    std::unique_ptr<StepTracer> tracer_;
    std::unique_ptr<GoalAllocator> goal_allocator_;  // lifelong mode only
    std::unique_ptr<PibtResolver> pibt_;             // pibt_corridor mode only
    std::vector<std::uint8_t> pibt_eligible_;
    std::vector<std::uint8_t> pibt_priority_class_;
    std::vector<CellId> pibt_forced_next_;
    std::vector<CellId> pibt_next_;
    std::vector<std::uint8_t> execution_failed_;
    std::vector<std::uint8_t> rescue_candidate_;   // pibt_corridor mode only
    std::vector<std::uint8_t> rescue_group_;
    std::vector<IntersectionId> rescue_member_;
    std::vector<std::uint8_t> scheduled_reserved_;
    std::vector<CellId> movement_origin_;
    std::vector<std::size_t> initial_route_lengths_;
    std::vector<std::uint64_t> completion_steps_;
    std::vector<IntersectionId> deadlock_queue_;
    std::vector<std::unordered_set<AgentId>> scheduled_members_;
    std::vector<bool> deadlock_waiting_;
    std::vector<int> intersection_available_;
    std::vector<int> intersection_capacity_;
    std::vector<std::uint8_t> deadlock_active_;
    std::vector<std::size_t> deadlock_priority_;

    [[nodiscard]] bool sample_command_failure(AgentId agent);

    // Per-timestep workspaces retain their capacity between calls to step().
    std::vector<std::size_t> inside_counts_;
    std::vector<std::size_t> prev_inside_counts_;
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
    void count_zone_entries(CellId current, CellId next);
    void rotate_blocked_cycles();
    void assign_lifelong_goals();
    void move_agent_to(Agent& agent, CellId next);
    [[nodiscard]] bool adjacent_or_equal(CellId current, CellId next) const;
    [[nodiscard]] CellId active_discharge_target(const Agent& agent) const;
    void run_pibt_movement();
    void compute_rescue_groups();
    std::vector<CellId> plan_global(CellId start, CellId goal);
};

}  // namespace lima
