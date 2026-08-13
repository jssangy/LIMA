#include "lima/simulation/simulator.hpp"

#include "lima/intersection/deadlock_detector.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace lima {
namespace {

std::optional<std::array<int, 4>> predict_final_stack_lengths(
    const std::array<int, 4>& arm_limits, const std::span<const IntersectionIntent> intents,
    const std::array<int, 4>& quotas) {
    std::array<int, 4> needs{};
    std::array<int, 4> lengths{};
    std::array<bool, 4> overflow{};
    for (const auto& intent : intents) {
        const auto d = static_cast<std::size_t>(intent.exit);
        if (d < 4 && arm_limits[d] > 0) ++needs[d];
    }
    for (std::size_t d = 0; d < 4; ++d) {
        const int capacity = arm_limits[d];
        overflow[d] = needs[d] > capacity;
        lengths[d] = overflow[d] ? capacity : needs[d];
    }
    for (std::size_t type = 0; type < 4; ++type) {
        int extra = overflow[type] ? needs[type] - arm_limits[type] : 0;
        while (extra-- > 0) {
            int destination = -1;
            for (std::size_t d = 0; d < 4; ++d) {
                if (d == type || overflow[d] || arm_limits[d] == 0
                    || lengths[d] >= arm_limits[d]) continue;
                if (destination < 0 || lengths[d] < lengths[static_cast<std::size_t>(destination)]) destination = static_cast<int>(d);
            }
            if (destination < 0) return std::nullopt;
            ++lengths[static_cast<std::size_t>(destination)];
        }
    }
    for (;;) {
        int source = -1;
        int largest_overflow = 0;
        for (std::size_t d = 0; d < 4; ++d) {
            const int excess = lengths[d] - quotas[d];
            if (excess > largest_overflow) {
                largest_overflow = excess;
                source = static_cast<int>(d);
            }
        }
        if (source < 0) break;
        int destination = -1;
        int best_slack = 0;
        for (std::size_t d = 0; d < 4; ++d) {
            if (static_cast<int>(d) == source || arm_limits[d] == 0) continue;
            const int slack = quotas[d] - lengths[d];
            if (slack > best_slack) {
                best_slack = slack;
                destination = static_cast<int>(d);
            }
        }
        if (destination < 0) return std::nullopt;
        --lengths[static_cast<std::size_t>(source)];
        ++lengths[static_cast<std::size_t>(destination)];
    }
    return lengths;
}

}  // namespace

Simulator::Simulator(GridMap map, const std::span<const Task> tasks, const PlannerKind planner_kind,
                     const std::uint64_t seed, SimulatorConfig config,
                     const std::span<const std::vector<Coord>> preset_routes)
    : map_(std::move(map)), config_(std::move(config)), rng_(seed),
      planner_(make_planner(planner_kind, map_, rng_)),
      topology_(IntersectionTopology::build(map_)),
      solver_(make_solver(config_.solver)), coordinator_(*solver_, config_.pibt_corridor),
      discharge_(config_.discharge) {
    agents_.reserve(tasks.size());
    std::unordered_set<CellId> occupied;
    // Sink semantics (adapted from the coworker despawn_at_goal ctor): on
    // maps with S cells, disappearing AMRs must target shared exits only
    // (an AMR is removed as soon as it arrives, so S cells may be shared).
    // Stay-at-goal AMRs park forever, so their goals must be unique.
    // All three benchmark maps carry S-only goal sets, so both rules hold
    // for every existing scenario.
    const std::unordered_set<CellId> sinks(map_.sink_cells().begin(), map_.sink_cells().end());
    std::unordered_set<CellId> assigned_goals;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const CellId start = map_.in_bounds(tasks[i].start) ? map_.cell(tasks[i].start) : kInvalidCell;
        const CellId goal = map_.in_bounds(tasks[i].goal) ? map_.cell(tasks[i].goal) : kInvalidCell;
        if (!map_.traversable(start) || !map_.traversable(goal)) throw std::runtime_error("task endpoint is not traversable");
        if (config_.goal_behavior == GoalBehavior::Disappear && !sinks.empty() && !sinks.contains(goal))
            throw std::runtime_error("maps with S cells require every task goal to be an S cell");
        if (config_.goal_behavior == GoalBehavior::Stay && !assigned_goals.insert(goal).second)
            throw std::runtime_error("persistent AMRs require unique task goal positions");
        if (!occupied.insert(start).second) throw std::runtime_error("duplicate task start position");
        std::vector<CellId> route;
        if (i < preset_routes.size() && !preset_routes[i].empty()) {
            route.reserve(preset_routes[i].size());
            for (const Coord c : preset_routes[i]) {
                if (!map_.traversable(c)) throw std::runtime_error("preset route leaves traversable space");
                if (!route.empty()) {
                    const Coord prev = map_.coord(route.back());
                    if (std::abs(prev.x - c.x) + std::abs(prev.y - c.y) > 1)
                        throw std::runtime_error("preset route is not 4-connected");
                }
                route.push_back(map_.cell(c));
            }
        } else {
            route = plan_global(start, goal);
        }
        if (route.empty() || route.front() != start || route.back() != goal) throw std::runtime_error("task has no route");
        Agent agent;
        agent.id = static_cast<AgentId>(i);
        agent.position = start;
        agent.goal = goal;
        agent.route = std::move(route);
        agents_.push_back(std::move(agent));
    }
    for (Agent& agent : agents_) {
        if (agent.position == agent.goal) {
            agent.active = false;
            ++stats_.completed;
        }
    }
    initial_route_lengths_.reserve(agents_.size());
    for (const Agent& agent : agents_) initial_route_lengths_.push_back(agent.route.size());
    completion_steps_.resize(agents_.size(), 0);
    if (!config_.metrics_dir.empty()) metrics_ = std::make_unique<MetricsCollector>(config_.metrics_dir);
    if (!config_.trace_path.empty()) {
        tracer_ = std::make_unique<StepTracer>(config_.trace_path, map_, config_.map_file, seed, agents_);
    }
    intersection_available_.resize(topology_.intersections().size());
    intersection_capacity_.resize(topology_.intersections().size());
    scheduled_members_.resize(topology_.intersections().size());
    deadlock_waiting_.resize(topology_.intersections().size(), false);
    deadlock_active_.resize(topology_.intersections().size(), 0);
    deadlock_priority_.resize(topology_.intersections().size(), 0);
    for (const Intersection& intersection : topology_.intersections()) {
        const auto iid = static_cast<std::size_t>(intersection.id);
        intersection_capacity_[iid] = scheduling_capacity(intersection, config_.isolation);
        intersection_available_[iid] = intersection_capacity_[iid];
    }
    for (const Agent& agent : agents_) if (agent.active) {
        for (const IntersectionId iid : topology_.memberships(agent.position))
            --intersection_available_[static_cast<std::size_t>(iid)];
    }

    const std::size_t intersection_count = topology_.intersections().size();
    inside_counts_.resize(intersection_count);
    members_.resize(intersection_count);
    intents_.resize(intersection_count);
    check_.resize(intersection_count);
    stalled_.resize(intersection_count);
    blocked_.resize(intersection_count);
    occupancy_.resize(static_cast<std::size_t>(map_.cell_count()), kNoAgent);
    normal_occupied_.resize(static_cast<std::size_t>(map_.cell_count()));
    candidates_.reserve(intersection_count);
    pending_.reserve(intersection_count);
    if (config_.goal_behavior == GoalBehavior::Lifelong) {
        // Dedicated task-stream RNG: goal arrivals must never consume rng_,
        // which drives planner tie-breaking for every mode.
        constexpr std::uint64_t task_seed_salt = 0x9e3779b97f4a7c15ULL;
        goal_allocator_ = std::make_unique<GoalAllocator>(map_, agents_, seed ^ task_seed_salt);
    }
    if (config_.pibt_corridor) {
        pibt_ = std::make_unique<PibtResolver>(map_, seed);
        pibt_eligible_.resize(agents_.size());
        pibt_priority_class_.resize(agents_.size());
        pibt_forced_next_.resize(agents_.size(), kInvalidCell);
        pibt_next_.resize(agents_.size(), kInvalidCell);
    }
}


bool Simulator::has_active_neighbor(const IntersectionId intersection) const {
    const auto& neighbors = topology_.intersections()[static_cast<std::size_t>(intersection)].neighbors;
    return std::any_of(neighbors.begin(), neighbors.end(), [&](const IntersectionId neighbor) {
        return neighbor >= 0 && deadlock_active_[static_cast<std::size_t>(neighbor)] != 0;
    });
}

void Simulator::rebuild_deadlock_priorities() {
    for (std::size_t priority = 0; priority < deadlock_queue_.size(); ++priority) {
        deadlock_priority_[static_cast<std::size_t>(deadlock_queue_[priority])] = priority;
    }
}

bool Simulator::block_intersection(const CellId current, const CellId next, const bool normal_only) const {
    const auto& current_memberships = topology_.memberships(current);
    const bool current_outside = current_memberships.empty();
    const bool current_center = topology_.is_center(current);
    if (!current_outside && !current_center) return false;

    if (!topology_.is_arm_tip(next)) return false;

    IntersectionId entering = -1;
    for (const IntersectionId iid : topology_.memberships(next)) {
        if (std::find(current_memberships.begin(), current_memberships.end(), iid) == current_memberships.end()) {
            entering = iid;
            break;
        }
    }
    if (entering < 0) return false;
    if (normal_only && intersection_available_[static_cast<std::size_t>(entering)] <= config_.isolation.hysteresis) return true;

    const auto priority = [&](const IntersectionId iid) {
        return deadlock_active_[static_cast<std::size_t>(iid)] != 0
            ? deadlock_priority_[static_cast<std::size_t>(iid)] : deadlock_queue_.size();
    };
    const std::size_t current_priority = current_outside || current_memberships.empty()
        ? deadlock_queue_.size() : priority(current_memberships.front());
    return priority(entering) < current_priority;
}

void Simulator::update_available_on_move(const CellId current, const CellId next) {
    const auto& from = topology_.memberships(current);
    const auto& to = topology_.memberships(next);
    for (const IntersectionId iid : from) {
        if (std::find(to.begin(), to.end(), iid) != to.end()) continue;
        const int capacity = intersection_capacity_[static_cast<std::size_t>(iid)];
        intersection_available_[static_cast<std::size_t>(iid)] = std::min(
            capacity, intersection_available_[static_cast<std::size_t>(iid)] + 1);
    }
    for (const IntersectionId iid : to) {
        if (std::find(from.begin(), from.end(), iid) != from.end()) continue;
        intersection_available_[static_cast<std::size_t>(iid)] = std::max(
            0, intersection_available_[static_cast<std::size_t>(iid)] - 1);
    }
}

void Simulator::count_zone_entries(const CellId current, const CellId next) {
    if (!metrics_ || current == next) return;
    const auto& from = topology_.memberships(current);
    int gained = 0;
    for (const IntersectionId iid : topology_.memberships(next)) {
        if (std::find(from.begin(), from.end(), iid) == from.end()) ++gained;
    }
    if (gained > 0) metrics_->add_acquisitions(gained);
}

void Simulator::insert_scheduled_path(Agent& agent, const ScheduledPath& scheduled,
                                      const IntersectionId intersection) {
    if (scheduled.path.size() < 2 || agent.route_cursor >= agent.route.size()) return;
    const CellId merge_point = scheduled.path.back();
    const CellId last_original = agent.route[agent.route_cursor];

    std::size_t exit_index = agent.route.size();
    for (std::size_t i = agent.route_cursor + 1; i < agent.route.size(); ++i) {
        if (agent.route[i] == scheduled.target_exit) {
            exit_index = i;
            break;
        }
    }
    const CellId rejoin = exit_index < agent.route.size() ? scheduled.target_exit : last_original;
    std::vector<CellId> continuation;
    if (exit_index < agent.route.size()) {
        continuation.assign(agent.route.begin() + static_cast<std::ptrdiff_t>(exit_index + 1), agent.route.end());
    } else {
        continuation.assign(agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor + 1), agent.route.end());
    }

    std::vector<CellId> bridge{merge_point};
    if (merge_point != rejoin) bridge = planner_->plan(merge_point, rejoin);
    if (bridge.empty()) return;

    std::vector<CellId> connected;
    connected.reserve(agent.route_cursor + scheduled.path.size() + bridge.size() + continuation.size());
    connected.insert(connected.end(), agent.route.begin(),
        agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor + 1));
    connected.insert(connected.end(), scheduled.path.begin() + 1, scheduled.path.end());
    if (bridge.size() > 1) connected.insert(connected.end(), bridge.begin() + 1, bridge.end());
    connected.insert(connected.end(), continuation.begin(), continuation.end());
    agent.route = std::move(connected);
    agent.scheduling_remaining = scheduled.path.size() - 1;
    agent.schedule_group = intersection;
}

void Simulator::move_agent(Agent& agent) {
    const CellId previous = agent.position;
    if (agent.route_cursor + 1 < agent.route.size()) {
        ++agent.route_cursor;
        agent.position = agent.route[agent.route_cursor];
    }
    if (agent.scheduling_remaining > 0) --agent.scheduling_remaining;
    if (previous != agent.position) {
        ++agent.moves;
        ++stats_.committed_moves;
    }
    agent.wait_steps = 0;
    agent.wait_reason = agent.scheduled() && previous == agent.position
        ? WaitReason::ScheduledHold : WaitReason::None;
}

bool Simulator::step() {
    if (done()) return false;
    const bool all_stalled = std::all_of(agents_.begin(), agents_.end(), [&](const Agent& agent) {
        return !agent.active || agent.wait_steps >= config_.stall_threshold;
    });
    if (all_stalled) return false;
    ++stats_.timestep;

    const std::size_t intersection_count = topology_.intersections().size();
    prev_inside_counts_ = inside_counts_;
    std::fill(inside_counts_.begin(), inside_counts_.end(), 0);
    std::fill(check_.begin(), check_.end(), false);
    std::fill(stalled_.begin(), stalled_.end(), false);
    for (auto& members : members_) members.clear();
    for (const Agent& agent : agents_) if (agent.active) {
        for (const IntersectionId iid : topology_.memberships(agent.position)) {
            ++inside_counts_[static_cast<std::size_t>(iid)];
            members_[static_cast<std::size_t>(iid)].push_back(agent.id);
        }
        for (const IntersectionId iid : topology_.memberships(agent.position)) {
            const Intersection& intersection = topology_.intersections()[static_cast<std::size_t>(iid)];
            if (agent.position == intersection.center || intersection.is_tip(agent.position))
                check_[static_cast<std::size_t>(iid)] = true;
        }
    }
    for (std::size_t iid = 0; iid < intersection_count; ++iid) {
        if (!members_[iid].empty()) {
            if (config_.discharge.partial_stall >= 1.0) {
                stalled_[iid] = std::all_of(members_[iid].begin(), members_[iid].end(), [&](const AgentId id) {
                    return agents_[static_cast<std::size_t>(id)].wait_steps >= 1;
                });
            } else {
                const std::size_t waiting = static_cast<std::size_t>(std::count_if(
                    members_[iid].begin(), members_[iid].end(), [&](const AgentId id) {
                        return agents_[static_cast<std::size_t>(id)].wait_steps >= 1;
                    }));
                stalled_[iid] = static_cast<double>(waiting)
                    >= config_.discharge.partial_stall * static_cast<double>(members_[iid].size());
            }
        }
        check_[iid] = check_[iid] || stalled_[iid] || deadlock_waiting_[iid];
    }
    if (config_.gate_resync) {
        // Ground-truth admission budget.  Quota reservations made below will
        // re-debit within the same step, so reservations survive while the
        // leaked (never-repaid) exit credits are restored.
        for (std::size_t iid = 0; iid < intersection_count; ++iid) {
            intersection_available_[iid] = intersection_capacity_[iid]
                - static_cast<int>(inside_counts_[iid]);
        }
    }

    bool queue_changed = false;
    for (auto queue = deadlock_queue_.begin(); queue != deadlock_queue_.end();) {
        const auto iid = static_cast<std::size_t>(*queue);
        auto& scheduled = scheduled_members_[iid];
        for (auto member = scheduled.begin(); member != scheduled.end();) {
            const Agent& agent = agents_[static_cast<std::size_t>(*member)];
            if (!agent.active || !agent.scheduled()) member = scheduled.erase(member);
            else ++member;
        }
        if (scheduled.empty()) {
            deadlock_active_[iid] = 0;
            queue = deadlock_queue_.erase(queue);
            queue_changed = true;
        } else ++queue;
    }
    if (queue_changed) rebuild_deadlock_priorities();

    candidates_.clear();
    for (const Intersection& intersection : topology_.intersections()) {
        const auto iid = static_cast<std::size_t>(intersection.id);
        if (!check_[iid] || deadlock_active_[iid] != 0) continue;
        auto& intersection_intents = intents_[iid];
        collect_intents(intersection, agents_, members_[iid], intersection_intents);
        if (intersection_intents.size() < 2 || !has_intersection_deadlock(intersection, intersection_intents)) {
            deadlock_waiting_[iid] = false;
            continue;
        }
        ++stats_.detected_deadlocks;
        if (has_active_neighbor(intersection.id)) {
            deadlock_waiting_[iid] = true;
            continue;
        }
        deadlock_waiting_[iid] = false;
        candidates_.push_back(intersection.id);
    }
    std::sort(candidates_.begin(), candidates_.end(), [&](const IntersectionId lhs, const IntersectionId rhs) {
        const auto l = inside_counts_[static_cast<std::size_t>(lhs)];
        const auto r = inside_counts_[static_cast<std::size_t>(rhs)];
        return l != r ? l < r : lhs < rhs;
    });

    pending_.clear();
    for (const IntersectionId iid_value : candidates_) {
        const auto iid = static_cast<std::size_t>(iid_value);
        if (has_active_neighbor(iid_value)) {
            deadlock_waiting_[iid] = true;
            continue;
        }
        const Intersection& intersection = topology_.intersections()[iid];
        std::array<int, 4> arm_limits{};
        for (std::size_t d = 0; d < 4; ++d) arm_limits[d] = static_cast<int>(intersection.arms[d].size());
        std::span<const IntersectionIntent> intersection_intents = intents_[iid];
        std::vector<IntersectionIntent> subset;
        if (inside_counts_[iid] > static_cast<std::size_t>(intersection_capacity_[iid])) {
            if (!config_.subset_scheduling) {
                deadlock_waiting_[iid] = true;
                continue;
            }
            // Saturated regime: solve for the innermost bound-many agents per
            // arm and treat everything deeper as a wall (zone-local knowledge).
            std::array<std::vector<std::pair<int, const IntersectionIntent*>>, 4> by_direction;
            const IntersectionIntent* center_intent = nullptr;
            for (const auto& intent : intents_[iid]) {
                if (intent.current == Direction::Center) {
                    center_intent = &intent;
                    continue;
                }
                const auto d = static_cast<std::size_t>(intent.current);
                if (d >= 4) continue;
                const auto& arm = intersection.arms[d];
                const auto found = std::find(arm.begin(), arm.end(), intent.position);
                if (found == arm.end()) continue;
                by_direction[d].emplace_back(static_cast<int>(found - arm.begin()), &intent);
            }
            for (auto& entries : by_direction) {
                std::sort(entries.begin(), entries.end(),
                          [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            }
            int budget = intersection_capacity_[iid] - (center_intent ? 1 : 0);
            std::array<int, 4> take{};
            bool progress = true;
            while (budget > 0 && progress) {
                progress = false;
                for (std::size_t d = 0; d < 4 && budget > 0; ++d) {
                    if (take[d] < static_cast<int>(by_direction[d].size())) {
                        ++take[d];
                        --budget;
                        progress = true;
                    }
                }
            }
            subset.clear();
            if (center_intent) subset.push_back(*center_intent);
            for (std::size_t d = 0; d < 4; ++d) {
                for (int k = 0; k < take[d]; ++k) subset.push_back(*by_direction[d][static_cast<std::size_t>(k)].second);
                arm_limits[d] = take[d] < static_cast<int>(by_direction[d].size())
                    ? by_direction[d][static_cast<std::size_t>(take[d])].first
                    : static_cast<int>(intersection.arms[d].size());
            }
            if (subset.size() < 2) {
                deadlock_waiting_[iid] = true;
                continue;
            }
            intersection_intents = subset;
        }
        std::array<int, 4> quotas{};
        std::array<int, 4> initial{};
        int quota_sum = 0;
        for (const auto& intent : intersection_intents) {
            const auto d = static_cast<std::size_t>(intent.current);
            if (d < 4) ++initial[d];
        }
        for (std::size_t d = 0; d < 4; ++d) {
            const int capacity = arm_limits[d];
            if (capacity == 0) continue;
            int neighbor_available = capacity;
            const IntersectionId neighbor = intersection.neighbors[d];
            if (neighbor >= 0) neighbor_available = std::max(
                0, intersection_available_[static_cast<std::size_t>(neighbor)]);
            quotas[d] = std::min(capacity, neighbor_available + initial[d]);
            quota_sum += quotas[d];
        }
        const auto final_lengths = predict_final_stack_lengths(arm_limits, intersection_intents, quotas);
        if (quota_sum < static_cast<int>(intersection_intents.size()) || !final_lengths) {
            deadlock_waiting_[iid] = true;
            continue;
        }
        for (std::size_t d = 0; d < 4; ++d) {
            const IntersectionId neighbor = intersection.neighbors[d];
            if (neighbor >= 0)
                intersection_available_[static_cast<std::size_t>(neighbor)] -= (*final_lengths)[d] - initial[d];
        }
        ScheduleTelemetry telemetry;
        auto plan = coordinator_.schedule(intersection, intersection_intents, quotas, arm_limits,
                                          metrics_ ? &telemetry : nullptr);
        if (metrics_) metrics_->on_solver_invocation(stats_.timestep, iid_value, telemetry, plan.has_value());
        if (!plan) {
            for (std::size_t d = 0; d < 4; ++d) {
                const IntersectionId neighbor = intersection.neighbors[d];
                if (neighbor >= 0)
                    intersection_available_[static_cast<std::size_t>(neighbor)] += (*final_lengths)[d] - initial[d];
            }
            deadlock_waiting_[iid] = true;
            continue;
        }
        deadlock_waiting_[iid] = false;
        deadlock_queue_.push_back(iid_value);
        deadlock_active_[iid] = 1;
        deadlock_priority_[iid] = deadlock_queue_.size() - 1;
        if (metrics_) {
            int neighbor_count = 0;
            for (const IntersectionId neighbor : intersection.neighbors) if (neighbor >= 0) ++neighbor_count;
            metrics_->add_gate_signals(neighbor_count);
        }
        pending_.push_back({iid_value, std::move(*plan)});
    }
    std::sort(pending_.begin(), pending_.end(), [](const PendingSchedule& lhs, const PendingSchedule& rhs) {
        return lhs.intersection < rhs.intersection;
    });
    const bool observe_schedules = metrics_ != nullptr || tracer_ != nullptr;
    for (auto& schedule : pending_) {
        std::vector<AgentId> applied;
        for (const ScheduledPath& path : schedule.paths) {
            Agent& agent = agents_[static_cast<std::size_t>(path.agent)];
            insert_scheduled_path(agent, path, schedule.intersection);
            if (agent.scheduled()) {
                scheduled_members_[static_cast<std::size_t>(schedule.intersection)].insert(agent.id);
                if (observe_schedules) applied.push_back(agent.id);
            }
        }
        if (metrics_) metrics_->add_broadcasts(static_cast<int>(applied.size()));
        if (tracer_ && !applied.empty()) tracer_->add_schedule(schedule.intersection, std::move(applied));
    }

    if (config_.discharge_enabled) {
        auto discharge_events = discharge_.run({topology_, agents_, members_, stalled_, deadlock_active_, *planner_, rng_,
                                                &intersection_available_, &prev_inside_counts_});
        for (auto& event : discharge_events) {
            if (metrics_) {
                metrics_->on_discharge(stats_.timestep, event.intersection, event.agent_ids, event.loop_cells);
                for (const AgentId id : event.agent_ids) metrics_->note_discharged_agent(id);
            }
            if (tracer_) tracer_->add_discharge(event.intersection, std::move(event.agent_ids));
        }
    }

    std::fill(occupancy_.begin(), occupancy_.end(), kNoAgent);
    for (const Agent& agent : agents_) if (agent.active)
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;

    if (config_.pibt_corridor) {
        run_pibt_movement();
    } else {
    for (Agent& agent : agents_) {
        if (!agent.active || agent.scheduled()) continue;
        const CellId next = agent.intended_cell();
        if (block_intersection(agent.position, next, true)) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::IntersectionCapacity;
            continue;
        }
        if (occupancy_[static_cast<std::size_t>(next)] == kNoAgent) {
            occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
            update_available_on_move(agent.position, next);
            count_zone_entries(agent.position, next);
            move_agent(agent);
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        } else {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::Dependency;
        }
    }
    }

    std::fill(normal_occupied_.begin(), normal_occupied_.end(), false);
    for (const Agent& agent : agents_) if (agent.active && !agent.scheduled())
        normal_occupied_[static_cast<std::size_t>(agent.position)] = true;
    std::fill(blocked_.begin(), blocked_.end(), false);
    for (const IntersectionId iid_value : deadlock_queue_) {
        const auto iid = static_cast<std::size_t>(iid_value);
        for (const AgentId id : scheduled_members_[iid]) {
            const Agent& agent = agents_[static_cast<std::size_t>(id)];
            if (!agent.active) continue;
            if (block_intersection(agent.position, agent.intended_cell(), false)
                || normal_occupied_[static_cast<std::size_t>(agent.intended_cell())]) {
                blocked_[iid] = true;
                break;
            }
        }
    }
    for (Agent& agent : agents_) {
        if (!agent.active || !agent.scheduled()) continue;
        const auto iid = static_cast<std::size_t>(agent.schedule_group);
        if (iid < blocked_.size() && blocked_[iid]) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::ScheduleGroup;
            continue;
        }
        occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
        const CellId previous = agent.position;
        move_agent(agent);
        count_zone_entries(previous, agent.position);
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
    }

    if (config_.rotation_enabled) rotate_blocked_cycles();

    for (Agent& agent : agents_) {
        if (!agent.active || agent.position != agent.goal) continue;
        if (config_.goal_behavior == GoalBehavior::Disappear) {
            agent.active = false;
            ++stats_.completed;
            completion_steps_[static_cast<std::size_t>(agent.id)] = stats_.timestep;
            if (tracer_) tracer_->add_completion(agent.id);
        } else if (config_.goal_behavior == GoalBehavior::Stay) {
            if (!agent.reached) {
                agent.reached = true;
                ++stats_.completed;
                completion_steps_[static_cast<std::size_t>(agent.id)] = stats_.timestep;
                if (tracer_) tracer_->add_completion(agent.id);
            }
        } else {  // Lifelong: serve the task; the allocator hands out the next one
            if (!agent.awaiting_goal) {
                agent.awaiting_goal = true;
                ++agent.tasks_completed;
                ++stats_.completed;
                completion_steps_[static_cast<std::size_t>(agent.id)] = stats_.timestep;
                if (tracer_) tracer_->add_completion(agent.id);
            }
        }
    }
    if (goal_allocator_) assign_lifelong_goals();
    if (metrics_) metrics_->flush_step(stats_.timestep);
    if (tracer_) tracer_->flush_step(stats_.timestep, agents_);
    return true;
}

void Simulator::assign_lifelong_goals() {
    for (Agent& agent : agents_) {
        // Reassignment is deferred while the agent still executes a scheduled
        // intersection path: replacing its route mid-schedule would corrupt
        // the coordinator's committed plan.  Deferred agents retry next step.
        if (!agent.active || !agent.awaiting_goal || agent.scheduled()) continue;
        std::vector<CellId> route;
        const auto next_goal = goal_allocator_->reassign(
            agent.id, agent.goal, agent.position, occupancy_, [&](const CellId candidate) {
                route = plan_global(agent.position, candidate);
                return !route.empty() && route.front() == agent.position && route.back() == candidate;
            });
        if (!next_goal) continue;
        agent.goal = *next_goal;
        agent.route = std::move(route);
        agent.route_cursor = 0;
        agent.wait_steps = 0;
        agent.wait_reason = WaitReason::None;
        agent.awaiting_goal = false;
    }
}

void Simulator::move_agent_to(Agent& agent, const CellId next) {
    const CellId previous = agent.position;
    if (next != previous) {
        // A PIBT sidestep does not replace the global route. Rejoin it when the
        // selected cell matches any future route cell; otherwise keep waiting
        // for the same next waypoint on subsequent timesteps.
        const auto begin = agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor + 1);
        const auto found = std::find(begin, agent.route.end(), next);
        if (found != agent.route.end())
            agent.route_cursor = static_cast<std::size_t>(std::distance(agent.route.begin(), found));
        agent.position = next;
        ++agent.moves;
        ++stats_.committed_moves;
    }
    agent.wait_steps = 0;
    agent.wait_reason = WaitReason::None;
}

bool Simulator::adjacent_or_equal(const CellId current, const CellId next) const {
    if (current == next) return true;
    const auto neighbors = map_.neighbors(current);
    return std::find(neighbors.begin(), neighbors.end(), next) != neighbors.end();
}

CellId Simulator::active_discharge_target(const Agent& agent) const {
    for (const IntersectionId iid_value : topology_.memberships(agent.position)) {
        const auto iid = static_cast<std::size_t>(iid_value);
        if (deadlock_active_[iid] == 0) continue;

        const Intersection& intersection = topology_.intersections()[iid];
        const Direction direction = intersection.direction_of(agent.position);
        const auto d = static_cast<std::size_t>(direction);
        if (d >= 4) continue;

        const auto& arm = intersection.arms[d];
        const auto current = std::find(arm.begin(), arm.end(), agent.position);
        if (current == arm.end()) continue;
        if (current + 1 != arm.end()) {
            const CellId outward = *(current + 1);
            return agent.intended_cell() == outward ? kInvalidCell : outward;
        }

        const Coord center = map_.coord(intersection.center);
        const Coord tip = map_.coord(agent.position);
        const Coord delta{
            tip.x == center.x ? 0 : (tip.x > center.x ? 1 : -1),
            tip.y == center.y ? 0 : (tip.y > center.y ? 1 : -1),
        };
        const Coord outside{tip.x + delta.x, tip.y + delta.y};
        if (map_.traversable(outside)) {
            const CellId outward = map_.cell(outside);
            return agent.intended_cell() == outward ? kInvalidCell : outward;
        }
    }
    return kInvalidCell;
}

// PIBT-corridor movement phase (opt-in): agents outside managed intersections
// move via priority-inheritance backtracking; agents inside keep the original
// single-route semantics except for constrained boundary-exit roots.
void Simulator::run_pibt_movement() {
    // Eligibility is captured before any movement. Normal outside agents use
    // PIBT freely. An unscheduled agent crossing from a managed intersection
    // to the outside joins only as a high-priority boundary-exit root, so it
    // can pass inherited priority to an outside blocker.
    std::fill(pibt_eligible_.begin(), pibt_eligible_.end(), 0);
    std::fill(pibt_priority_class_.begin(), pibt_priority_class_.end(), 0);
    std::fill(pibt_forced_next_.begin(), pibt_forced_next_.end(), kInvalidCell);
    for (const Agent& agent : agents_) {
        if (!agent.active || agent.scheduled()) continue;
        const auto index = static_cast<std::size_t>(agent.id);
        const CellId discharge = active_discharge_target(agent);
        if (discharge != kInvalidCell) {
            pibt_eligible_[index] = 1;
            pibt_priority_class_[index] = 2;
            pibt_forced_next_[index] = discharge;
            continue;
        }
        // A PIBT detour can leave the AMR next to, but not directly on, its
        // current route waypoint. Keep that recovery atomic and local instead
        // of letting the normal phase jump to a non-adjacent future cell.
        if (!adjacent_or_equal(agent.position, agent.intended_cell())) {
            pibt_eligible_[index] = 1;
            continue;
        }
        const bool current_outside = topology_.memberships(agent.position).empty();
        const bool next_outside = topology_.memberships(agent.intended_cell()).empty();
        if (current_outside) {
            pibt_eligible_[index] = 1;
        } else if (next_outside) {
            pibt_eligible_[index] = 1;
            pibt_priority_class_[index] = 1;
        }
    }

    // Keep the original single-route movement semantics inside managed
    // intersections, except for the constrained boundary-exit roots selected
    // above.
    for (Agent& agent : agents_) {
        if (!agent.active || agent.scheduled()
            || pibt_eligible_[static_cast<std::size_t>(agent.id)] != 0) continue;
        const CellId next = agent.intended_cell();
        if (!adjacent_or_equal(agent.position, next)) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::Dependency;
            continue;
        }
        // A route can contain explicit wait frames left by an intersection
        // schedule. Once the coordinated portion has ended, consuming that
        // frame must advance only the route cursor; treating the robot as the
        // occupant of its own destination leaves it waiting forever.
        if (next == agent.position && agent.route_cursor + 1 < agent.route.size()) {
            move_agent(agent);
            continue;
        }
        if (agent.schedule_group >= 0
            && deadlock_active_[static_cast<std::size_t>(agent.schedule_group)] != 0
            && std::find(topology_.memberships(next).begin(), topology_.memberships(next).end(),
                         agent.schedule_group) != topology_.memberships(next).end()
            && std::find(topology_.memberships(agent.position).begin(), topology_.memberships(agent.position).end(),
                         agent.schedule_group) == topology_.memberships(agent.position).end()) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::IntersectionReserved;
            continue;
        }
        if (block_intersection(agent.position, next, true)) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::IntersectionCapacity;
            continue;
        }
        if (occupancy_[static_cast<std::size_t>(next)] == kNoAgent) {
            occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
            update_available_on_move(agent.position, next);
            count_zone_entries(agent.position, next);
            move_agent(agent);
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        } else {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::Dependency;
        }
    }

    pibt_->resolve(agents_, occupancy_, pibt_eligible_, pibt_priority_class_,
        [&](const AgentId id, const CellId candidate) {
            const Agent& agent = agents_[static_cast<std::size_t>(id)];
            const CellId forced = pibt_forced_next_[static_cast<std::size_t>(id)];
            if (forced != kInvalidCell) return candidate == forced;
            if (candidate == agent.position) return true;
            if (agent.schedule_group >= 0
                && deadlock_active_[static_cast<std::size_t>(agent.schedule_group)] != 0) {
                const auto& memberships = topology_.memberships(candidate);
                if (std::find(memberships.begin(), memberships.end(), agent.schedule_group) != memberships.end())
                    return false;
            }
            if (pibt_priority_class_[static_cast<std::size_t>(id)] != 0)
                return candidate == agent.intended_cell();
            const bool candidate_outside = topology_.memberships(candidate).empty();
            const bool rejoins_current_waypoint = agent.route_cursor < agent.route.size()
                && candidate == agent.route[agent.route_cursor];
            // A detour may not enter a managed intersection. Crossing the
            // boundary is allowed only through the agent's global route and
            // remains subject to the existing intersection admission gate.
            if (!candidate_outside && candidate != agent.intended_cell()
                && !rejoins_current_waypoint) return false;
            return !block_intersection(agent.position, candidate, true);
        }, pibt_next_);

    for (Agent& agent : agents_) {
        const auto index = static_cast<std::size_t>(agent.id);
        if (pibt_eligible_[index] == 0) continue;
        const CellId next = pibt_next_[index];
        if (next == kInvalidCell || next == agent.position) {
            if (next == agent.position && agent.intended_cell() == agent.position
                && agent.route_cursor + 1 < agent.route.size()) {
                move_agent(agent);
                continue;
            }
            if ((agent.reached || agent.awaiting_goal) && agent.position == agent.goal) {
                agent.wait_steps = 0;
                agent.wait_reason = WaitReason::None;
                continue;
            }
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::Dependency;
            continue;
        }
        const CellId forced = pibt_forced_next_[index];
        if (forced != kInvalidCell && next == forced) {
            // Discharge temporarily moves the AMR one cell away from its
            // immutable global route. Insert the cell being left as an
            // explicit return waypoint. Without this handoff, route_cursor
            // still points at that cell and the following intended cell can
            // be two cells away, causing either a teleport or a permanent
            // PIBT wait at an intersection boundary.
            const auto insertion = agent.route.begin()
                + static_cast<std::ptrdiff_t>(agent.route_cursor + 1);
            if (insertion == agent.route.end() || *insertion != agent.position)
                agent.route.insert(insertion, agent.position);
        }
        update_available_on_move(agent.position, next);
        count_zone_entries(agent.position, next);
        move_agent_to(agent, next);
    }

    // Rebuild after the atomic PIBT commit. Clearing and setting in one agent
    // loop would corrupt rotations because another agent may enter a cell that
    // is cleared later in that same loop.
    std::fill(occupancy_.begin(), occupancy_.end(), kNoAgent);
    for (const Agent& agent : agents_) if (agent.active)
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
}

void Simulator::rotate_blocked_cycles() {
    std::vector<std::uint8_t> visited(agents_.size(), 0);
    std::vector<AgentId> chain;
    for (const Agent& seed : agents_) {
        if (!seed.active || seed.scheduled() || seed.wait_steps == 0) continue;
        if (visited[static_cast<std::size_t>(seed.id)]) continue;
        chain.clear();
        AgentId current = seed.id;
        std::size_t cycle_start = chain.max_size();
        while (true) {
            const Agent& agent = agents_[static_cast<std::size_t>(current)];
            if (visited[static_cast<std::size_t>(current)]) {
                const auto found = std::find(chain.begin(), chain.end(), current);
                if (found != chain.end()) cycle_start = static_cast<std::size_t>(found - chain.begin());
                break;
            }
            visited[static_cast<std::size_t>(current)] = 1;
            chain.push_back(current);
            // Only unscheduled agents that waited this step participate.
            if (!agent.active || agent.scheduled() || agent.wait_steps == 0) break;
            const CellId next = agent.intended_cell();
            if (next == agent.position) break;
            const AgentId occupant = occupancy_[static_cast<std::size_t>(next)];
            if (occupant == kNoAgent) break;
            current = occupant;
        }
        if (cycle_start == chain.max_size()) continue;
        const std::size_t length = chain.size() - cycle_start;
        if (length < 3) continue;  // a 2-cycle would be a forbidden swap
        // Cyclic permutation: every member steps into the cell its successor
        // vacates in the same timestep.
        for (std::size_t k = cycle_start; k < chain.size(); ++k) {
            Agent& agent = agents_[static_cast<std::size_t>(chain[k])];
            const CellId previous = agent.position;
            update_available_on_move(previous, agent.intended_cell());
            count_zone_entries(previous, agent.intended_cell());
            occupancy_[static_cast<std::size_t>(previous)] = kNoAgent;
            move_agent(agent);
        }
        for (std::size_t k = cycle_start; k < chain.size(); ++k) {
            const Agent& agent = agents_[static_cast<std::size_t>(chain[k])];
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        }
    }
}

std::string Simulator::check_invariants() const {
    std::unordered_set<CellId> seen;
    for (const Agent& agent : agents_) {
        if (!agent.active) continue;
        std::ostringstream problem;
        if (!map_.traversable(agent.position)) {
            problem << "agent " << agent.id << " stands on non-traversable cell " << agent.position;
            return problem.str();
        }
        if (!seen.insert(agent.position).second) {
            problem << "vertex conflict at cell " << agent.position << " involving agent " << agent.id;
            return problem.str();
        }
        if (agent.route_cursor >= agent.route.size() || agent.route[agent.route_cursor] != agent.position) {
            problem << "agent " << agent.id << " desynchronized from its route cursor";
            return problem.str();
        }
    }
    for (std::size_t iid = 0; iid < scheduled_members_.size(); ++iid) {
        if (deadlock_active_[iid] == 0 && !scheduled_members_[iid].empty()) {
            std::ostringstream problem;
            problem << "intersection " << iid << " holds scheduled members while inactive";
            return problem.str();
        }
    }
    return {};
}

void Simulator::write_metrics() {
    if (metrics_) metrics_->finalize(agents_, initial_route_lengths_, completion_steps_);
    if (tracer_) tracer_->finish();
}

std::vector<CellId> Simulator::plan_global(const CellId start, const CellId goal) {
    if (start == goal) return {start};
    if (config_.direct_routing) return planner_->plan(start, goal);
    const Coord source = map_.coord(start);
    const Coord destination = map_.coord(goal);
    std::vector<int> center_xs;
    std::vector<int> center_ys;
    for (const Intersection& intersection : topology_.intersections()) {
        const Coord c = map_.coord(intersection.center);
        center_xs.push_back(c.x);
        center_ys.push_back(c.y);
    }
    std::sort(center_xs.begin(), center_xs.end());
    center_xs.erase(std::unique(center_xs.begin(), center_xs.end()), center_xs.end());
    std::sort(center_ys.begin(), center_ys.end());
    center_ys.erase(std::unique(center_ys.begin(), center_ys.end()), center_ys.end());

    const auto in_range = [](const int value, const int a, const int b) {
        return value >= std::min(a, b) && value <= std::max(a, b);
    };
    const auto nearest = [&](const std::vector<int>& values, const int a, const int b, const int target) -> std::optional<int> {
        std::optional<int> best;
        for (const int value : values) if (in_range(value, a, b)) {
            if (!best || std::abs(value - target) < std::abs(*best - target)) best = value;
        }
        if (!best) for (const int value : values) {
            if (!best || std::abs(value - target) < std::abs(*best - target)) best = value;
        }
        return best;
    };
    const auto candidates = [&](const std::vector<int>& values, const int a, const int b) {
        std::vector<int> result;
        for (const int value : values) if (in_range(value, a, b)) result.push_back(value);
        return result;
    };
    const auto via = [&](const std::vector<Coord>& waypoints) -> std::vector<CellId> {
        CellId current = start;
        std::vector<CellId> route{start};
        for (const Coord waypoint : waypoints) {
            if (!map_.traversable(waypoint)) return {};
            const CellId target = map_.cell(waypoint);
            if (target == current) continue;
            auto segment = planner_->plan(current, target);
            if (segment.empty() || segment.back() != target) return {};
            route.insert(route.end(), segment.begin() + 1, segment.end());
            current = target;
        }
        return route;
    };

    bool vertical_goal = destination.y == 0 || destination.y + 1 == map_.height();
    bool horizontal_goal = destination.x == 0 || destination.x + 1 == map_.width();
    if (!vertical_goal && !horizontal_goal) horizontal_goal = true;

    if (vertical_goal) {
        const auto x_align = nearest(center_xs, source.x, destination.x, source.x);
        const auto choices = candidates(center_ys, source.y, destination.y);
        for (int attempt = 0; x_align && !choices.empty() && attempt < 8; ++attempt) {
            std::uniform_int_distribution<std::size_t> choose(0, choices.size() - 1);
            const int y = choices[choose(rng_)];
            auto route = via({{*x_align, source.y}, {*x_align, y}, {destination.x, y}, destination});
            if (!route.empty()) return route;
        }
    } else if (horizontal_goal) {
        const auto y_align = nearest(center_ys, source.y, destination.y, source.y);
        const auto choices = candidates(center_xs, source.x, destination.x);
        for (int attempt = 0; y_align && !choices.empty() && attempt < 8; ++attempt) {
            std::uniform_int_distribution<std::size_t> choose(0, choices.size() - 1);
            const int x = choices[choose(rng_)];
            auto route = via({{source.x, *y_align}, {x, *y_align}, {x, destination.y}, destination});
            if (!route.empty()) return route;
        }
    }
    return planner_->plan(start, goal);
}

}  // namespace lima
