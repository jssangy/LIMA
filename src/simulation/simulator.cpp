#include "lima/simulation/simulator.hpp"

#include "lima/intersection/deadlock_detector.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_set>

namespace lima {
namespace {

std::size_t scheduling_capacity(const Intersection& intersection) {
    std::size_t total = 0;
    std::size_t longest = 0;
    for (const auto& arm : intersection.arms) {
        total += arm.size();
        longest = std::max(longest, arm.size());
    }
    const bool isolated = std::none_of(
        intersection.neighbors.begin(), intersection.neighbors.end(),
        [](const IntersectionId neighbor) { return neighbor >= 0; });
    // With no downstream intersection there is no neighbor-discharge space
    // to reserve. All arm slots can safely participate in the local schedule.
    if (isolated) return total;
    return total - longest;
}

std::optional<std::array<int, 4>> predict_final_stack_lengths(
    const Intersection& intersection, const std::span<const IntersectionIntent> intents,
    const std::array<int, 4>& quotas) {
    std::array<int, 4> needs{};
    std::array<int, 4> lengths{};
    std::array<bool, 4> overflow{};
    for (const auto& intent : intents) {
        const auto d = static_cast<std::size_t>(intent.exit);
        if (d < 4 && !intersection.arms[d].empty()) ++needs[d];
    }
    for (std::size_t d = 0; d < 4; ++d) {
        const int capacity = static_cast<int>(intersection.arms[d].size());
        overflow[d] = needs[d] > capacity;
        lengths[d] = overflow[d] ? capacity : needs[d];
    }
    for (std::size_t type = 0; type < 4; ++type) {
        int extra = overflow[type] ? needs[type] - static_cast<int>(intersection.arms[type].size()) : 0;
        while (extra-- > 0) {
            int destination = -1;
            for (std::size_t d = 0; d < 4; ++d) {
                if (d == type || overflow[d] || intersection.arms[d].empty()
                    || lengths[d] >= static_cast<int>(intersection.arms[d].size())) continue;
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
            if (static_cast<int>(d) == source || intersection.arms[d].empty()) continue;
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
                     const std::uint64_t seed, const WorkloadMode workload)
    : map_(std::move(map)), rng_(seed), planner_(make_planner(planner_kind, map_, rng_)),
      topology_(IntersectionTopology::build(map_)), workload_(workload),
      despawn_at_goal_(!map_.sink_cells().empty()), pibt_(map_, seed) {
    agents_.reserve(tasks.size());
    std::unordered_set<CellId> occupied;
    std::unordered_set<CellId> assigned_goals;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const CellId start = map_.in_bounds(tasks[i].start) ? map_.cell(tasks[i].start) : kInvalidCell;
        const CellId goal = map_.in_bounds(tasks[i].goal) ? map_.cell(tasks[i].goal) : kInvalidCell;
        if (!map_.traversable(start) || !map_.traversable(goal)) throw std::runtime_error("task endpoint is not traversable");
        if (despawn_at_goal_
            && std::find(map_.sink_cells().begin(), map_.sink_cells().end(), goal) == map_.sink_cells().end())
            throw std::runtime_error("maps with S cells require every task goal to be an S cell");
        if (!occupied.insert(start).second) throw std::runtime_error("duplicate task start position");
        if (!despawn_at_goal_ && !assigned_goals.insert(goal).second)
            throw std::runtime_error("persistent AMRs require unique task goal positions");
        auto route = plan_global(start, goal);
        if (route.empty() || route.front() != start || route.back() != goal) throw std::runtime_error("task has no route");
        Agent agent;
        agent.id = static_cast<AgentId>(i);
        agent.position = start;
        agent.goal = goal;
        agent.route = std::move(route);
        agent.completed = start == goal;
        if (agent.completed) {
            if (despawn_at_goal_) agent.active = false;
            if (workload_ == WorkloadMode::OneShot) ++stats_.completed;
            else {
                agent.awaiting_goal = true;
                ++agent.tasks_completed;
                ++stats_.completed_tasks;
            }
        }
        agents_.push_back(std::move(agent));
    }
    intersection_available_.resize(topology_.intersections().size());
    intersection_capacity_.resize(topology_.intersections().size());
    scheduled_members_.resize(topology_.intersections().size());
    deadlock_waiting_.resize(topology_.intersections().size(), false);
    deadlock_active_.resize(topology_.intersections().size(), 0);
    deadlock_release_grace_.resize(topology_.intersections().size(), 0);
    deadlock_priority_.resize(topology_.intersections().size(), 0);
    for (const Intersection& intersection : topology_.intersections()) {
        const auto iid = static_cast<std::size_t>(intersection.id);
        intersection_capacity_[iid] = static_cast<int>(scheduling_capacity(intersection));
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
    intent_valid_.resize(intersection_count);
    schedule_decision_.resize(intersection_count);
    debug_initial_counts_.resize(intersection_count);
    debug_quotas_.resize(intersection_count);
    debug_final_counts_.resize(intersection_count);
    debug_quota_valid_.resize(intersection_count);
    debug_final_valid_.resize(intersection_count);
    check_.resize(intersection_count);
    stalled_.resize(intersection_count);
    blocked_.resize(intersection_count);
    occupancy_.resize(static_cast<std::size_t>(map_.cell_count()), kNoAgent);
    normal_occupied_.resize(static_cast<std::size_t>(map_.cell_count()));
    scheduled_reserved_.resize(static_cast<std::size_t>(map_.cell_count()));
    rescue_candidate_.resize(intersection_count);
    rescue_group_.resize(intersection_count);
    rescue_member_.resize(agents_.size());
    movement_origin_.resize(agents_.size(), kInvalidCell);
    movement_intended_.resize(agents_.size(), kInvalidCell);
    movement_scheduling_.resize(agents_.size());
    movement_wait_steps_.resize(agents_.size());
    pibt_eligible_.resize(agents_.size());
    pibt_priority_class_.resize(agents_.size());
    pibt_forced_next_.resize(agents_.size(), kInvalidCell);
    pibt_next_.resize(agents_.size(), kInvalidCell);
    candidates_.reserve(intersection_count);
    pending_.reserve(intersection_count);
    if (workload_ == WorkloadMode::Lifelong && !despawn_at_goal_) {
        constexpr std::uint64_t task_seed_salt = 0x9e3779b97f4a7c15ULL;
        goal_allocator_ = std::make_unique<GoalAllocator>(map_, agents_, seed ^ task_seed_salt);
        assign_lifelong_goals();
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
    if (normal_only && intersection_available_[static_cast<std::size_t>(entering)] <= 0) return true;

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

bool Simulator::step() {
    if (done()) return false;
    const bool all_stalled = std::all_of(agents_.begin(), agents_.end(), [&](const Agent& agent) {
        return !agent.active
            || (workload_ == WorkloadMode::OneShot && agent.completed)
            || agent.wait_steps >= 10;
    });
    if (all_stalled) return false;
    ++stats_.timestep;

    const std::size_t intersection_count = topology_.intersections().size();
    std::fill(inside_counts_.begin(), inside_counts_.end(), 0);
    std::fill(check_.begin(), check_.end(), false);
    std::fill(stalled_.begin(), stalled_.end(), false);
    std::fill(intent_valid_.begin(), intent_valid_.end(), 0);
    std::fill(schedule_decision_.begin(), schedule_decision_.end(), ScheduleDecision::NotChecked);
    std::fill(debug_quota_valid_.begin(), debug_quota_valid_.end(), 0);
    std::fill(debug_final_valid_.begin(), debug_final_valid_.end(), 0);
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
    // Rebuild capacity from the actual occupancy at the start of every
    // timestep.  A schedule's predicted final stack is only a reservation
    // while the schedules for this timestep are selected; carrying that
    // reservation into later timesteps makes it accumulate every time the
    // same unresolved deadlock is scheduled again.  Actual moves below may
    // still update this working counter for the remainder of the timestep,
    // but the next step always starts from the physical state again.
    for (std::size_t iid = 0; iid < intersection_count; ++iid) {
        intersection_available_[iid] = std::max(
            0, intersection_capacity_[iid] - static_cast<int>(inside_counts_[iid]));
    }
    for (std::size_t iid = 0; iid < intersection_count; ++iid) {
        if (!members_[iid].empty()) {
            stalled_[iid] = std::all_of(members_[iid].begin(), members_[iid].end(), [&](const AgentId id) {
                return agents_[static_cast<std::size_t>(id)].wait_steps >= 1;
            });
            // A two-agent edge swap can persist while other members keep this
            // intersection from becoming globally stalled. Detect that exact
            // event directly instead of broadening the trigger to every
            // partially waiting intersection.
            for (std::size_t i = 0; !check_[iid] && i < members_[iid].size(); ++i) {
                const Agent& a = agents_[static_cast<std::size_t>(members_[iid][i])];
                for (std::size_t j = i + 1; j < members_[iid].size(); ++j) {
                    const Agent& b = agents_[static_cast<std::size_t>(members_[iid][j])];
                    if (a.position != a.intended_cell()
                        && a.intended_cell() == b.position && b.intended_cell() == a.position) {
                        check_[iid] = true;
                        break;
                    }
                }
            }
        }
        check_[iid] = check_[iid] || stalled_[iid] || deadlock_waiting_[iid];
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
        if (scheduled.empty() && deadlock_release_grace_[iid] == 0) {
            // Keep ownership for one final movement phase. Released cohort
            // members still on an arm are pushed outward by PIBT before this
            // intersection can be scheduled again.
            deadlock_release_grace_[iid] = 1;
            ++queue;
        } else if (scheduled.empty()) {
            deadlock_release_grace_[iid] = 0;
            deadlock_active_[iid] = 0;
            queue = deadlock_queue_.erase(queue);
            queue_changed = true;
        } else {
            deadlock_release_grace_[iid] = 0;
            ++queue;
        }
    }
    if (queue_changed) rebuild_deadlock_priorities();

    candidates_.clear();
    for (const Intersection& intersection : topology_.intersections()) {
        const auto iid = static_cast<std::size_t>(intersection.id);
        if (!check_[iid]) continue;
        if (deadlock_active_[iid] != 0) {
            schedule_decision_[iid] = ScheduleDecision::Active;
            continue;
        }
        auto& intersection_intents = intents_[iid];
        collect_intents(intersection, agents_, members_[iid], intersection_intents);
        intent_valid_[iid] = 1;
        if (intersection_intents.size() < 2 || !has_intersection_deadlock(intersection, intersection_intents)) {
            schedule_decision_[iid] = ScheduleDecision::NoDeadlock;
            deadlock_waiting_[iid] = false;
            continue;
        }
        ++stats_.detected_deadlocks;
        if (has_active_neighbor(intersection.id)) {
            schedule_decision_[iid] = ScheduleDecision::NeighborActive;
            deadlock_waiting_[iid] = true;
            continue;
        }
        deadlock_waiting_[iid] = false;
        schedule_decision_[iid] = ScheduleDecision::Candidate;
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
            schedule_decision_[iid] = ScheduleDecision::NeighborActive;
            deadlock_waiting_[iid] = true;
            continue;
        }
        const Intersection& intersection = topology_.intersections()[iid];
        if (inside_counts_[iid] > static_cast<std::size_t>(intersection_capacity_[iid])) {
            schedule_decision_[iid] = ScheduleDecision::CapacityExceeded;
            deadlock_waiting_[iid] = true;
            continue;
        }
        const auto& intersection_intents = intents_[iid];
        std::array<int, 4> quotas{};
        std::array<int, 4> initial{};
        int quota_sum = 0;
        for (const auto& intent : intersection_intents) {
            const auto d = static_cast<std::size_t>(intent.current);
            if (d < 4) ++initial[d];
        }
        for (std::size_t d = 0; d < 4; ++d) {
            const int capacity = static_cast<int>(intersection.arms[d].size());
            if (capacity == 0) continue;
            int neighbor_available = capacity;
            const IntersectionId neighbor = intersection.neighbors[d];
            if (neighbor >= 0) neighbor_available = std::max(
                0, intersection_available_[static_cast<std::size_t>(neighbor)]);
            quotas[d] = std::min(capacity, neighbor_available + initial[d]);
            quota_sum += quotas[d];
        }
        debug_initial_counts_[iid] = initial;
        debug_quotas_[iid] = quotas;
        debug_quota_valid_[iid] = 1;
        const auto final_lengths = predict_final_stack_lengths(intersection, intersection_intents, quotas);
        if (quota_sum < static_cast<int>(intersection_intents.size()) || !final_lengths) {
            schedule_decision_[iid] = ScheduleDecision::NeighborQuotaBlocked;
            deadlock_waiting_[iid] = true;
            continue;
        }
        debug_final_counts_[iid] = *final_lengths;
        debug_final_valid_[iid] = 1;
        for (std::size_t d = 0; d < 4; ++d) {
            const IntersectionId neighbor = intersection.neighbors[d];
            if (neighbor >= 0)
                intersection_available_[static_cast<std::size_t>(neighbor)] -= (*final_lengths)[d] - initial[d];
        }
        auto plan = coordinator_.schedule(intersection, intersection_intents, quotas);
        if (!plan) {
            for (std::size_t d = 0; d < 4; ++d) {
                const IntersectionId neighbor = intersection.neighbors[d];
                if (neighbor >= 0)
                    intersection_available_[static_cast<std::size_t>(neighbor)] += (*final_lengths)[d] - initial[d];
            }
            schedule_decision_[iid] = ScheduleDecision::PlannerFailed;
            deadlock_waiting_[iid] = true;
            continue;
        }
        // Every participant belongs to this cohort even when its generated
        // path is omitted or trimmed for early discharge. schedule_group is
        // retained as a handoff token until this intersection becomes idle.
        for (const auto& intent : intersection_intents)
            agents_[static_cast<std::size_t>(intent.agent)].schedule_group = iid_value;
        deadlock_waiting_[iid] = false;
        schedule_decision_[iid] = ScheduleDecision::Activated;
        deadlock_queue_.push_back(iid_value);
        deadlock_active_[iid] = 1;
        deadlock_priority_[iid] = deadlock_queue_.size() - 1;
        pending_.push_back({iid_value, std::move(*plan)});
    }
    std::sort(pending_.begin(), pending_.end(), [](const PendingSchedule& lhs, const PendingSchedule& rhs) {
        return lhs.intersection < rhs.intersection;
    });
    for (auto& schedule : pending_) {
        for (const ScheduledPath& path : schedule.paths) {
            Agent& agent = agents_[static_cast<std::size_t>(path.agent)];
            insert_scheduled_path(agent, path, schedule.intersection);
            if (agent.scheduled()) scheduled_members_[static_cast<std::size_t>(schedule.intersection)].insert(agent.id);
        }
    }

    recover_stalled_intersections(members_, stalled_);

    std::fill(occupancy_.begin(), occupancy_.end(), kNoAgent);
    for (const Agent& agent : agents_) if (agent.active) {
        const auto index = static_cast<std::size_t>(agent.id);
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        movement_origin_[index] = agent.position;
        movement_intended_[index] = agent.intended_cell();
        movement_scheduling_[index] = agent.scheduling_remaining;
        movement_wait_steps_[index] = agent.wait_steps;
    }

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
            move_agent(agent);
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        } else {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::Dependency;
        }
    }

    pibt_.resolve(agents_, occupancy_, pibt_eligible_, pibt_priority_class_,
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
            if (agent.completed && agent.position == agent.goal) {
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
        move_agent_to(agent, next);
    }

    // Rebuild after the atomic PIBT commit. Clearing and setting in one agent
    // loop would corrupt rotations because another agent may enter a cell that
    // is cleared later in that same loop.
    std::fill(occupancy_.begin(), occupancy_.end(), kNoAgent);
    for (const Agent& agent : agents_) if (agent.active)
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;

    std::fill(normal_occupied_.begin(), normal_occupied_.end(), false);
    for (const Agent& agent : agents_) if (agent.active && !agent.scheduled())
        normal_occupied_[static_cast<std::size_t>(agent.position)] = true;
    std::fill(blocked_.begin(), blocked_.end(), false);
    std::fill(rescue_candidate_.begin(), rescue_candidate_.end(), 0);
    std::fill(rescue_group_.begin(), rescue_group_.end(), 0);
    std::fill(rescue_member_.begin(), rescue_member_.end(), kNoGroup);
    std::fill(scheduled_reserved_.begin(), scheduled_reserved_.end(), 0);

    // Preserve the normal group-wide scheduling rule. Merely record the
    // special failure mode where a scheduled destination is occupied by an
    // already released member of the same schedule.
    for (const IntersectionId iid_value : deadlock_queue_) {
        const auto iid = static_cast<std::size_t>(iid_value);
        for (const AgentId id : scheduled_members_[iid]) {
            const Agent& agent = agents_[static_cast<std::size_t>(id)];
            if (!agent.active) continue;
            const CellId next = agent.intended_cell();
            if (block_intersection(agent.position, next, false)) {
                blocked_[iid] = true;
                break;
            }
            if (!normal_occupied_[static_cast<std::size_t>(next)]) continue;
            const AgentId occupant = occupancy_[static_cast<std::size_t>(next)];
            if (occupant == kNoAgent) continue;
            const Agent& dependency = agents_[static_cast<std::size_t>(occupant)];
            if (!dependency.scheduled() && dependency.schedule_group == iid_value) {
                rescue_candidate_[iid] = 1;
            } else {
                blocked_[iid] = true;
                break;
            }
        }
        if (!blocked_[iid] && !rescue_candidate_[iid]) {
            for (const AgentId id : scheduled_members_[iid]) {
                const Agent& agent = agents_[static_cast<std::size_t>(id)];
                if (agent.active)
                    scheduled_reserved_[static_cast<std::size_t>(agent.intended_cell())] = 1;
            }
        }
    }

    // Rescue only a schedule cohort affected by the mixed
    // scheduled/released standstill. All unaffected intersections keep the
    // original movement semantics and are not made artificially atomic.
    for (const IntersectionId iid_value : deadlock_queue_) {
        const auto iid = static_cast<std::size_t>(iid_value);
        if (blocked_[iid] || !rescue_candidate_[iid]) continue;

        std::vector<AgentId> members;
        members.reserve(scheduled_members_[iid].size() + 4);
        for (const AgentId id : scheduled_members_[iid]) {
            const Agent& agent = agents_[static_cast<std::size_t>(id)];
            if (!agent.active || !agent.scheduled()) continue;
            rescue_member_[static_cast<std::size_t>(id)] = iid_value;
            members.push_back(id);
        }

        const auto next_for = [&](const Agent& agent) {
            const CellId forced = pibt_forced_next_[static_cast<std::size_t>(agent.id)];
            return !agent.scheduled() && forced != kInvalidCell ? forced : agent.intended_cell();
        };
        for (std::size_t cursor = 0; cursor < members.size(); ++cursor) {
            const Agent& agent = agents_[static_cast<std::size_t>(members[cursor])];
            const AgentId occupant = occupancy_[static_cast<std::size_t>(next_for(agent))];
            if (occupant == kNoAgent || occupant == agent.id
                || rescue_member_[static_cast<std::size_t>(occupant)] == iid_value) continue;
            const Agent& dependency = agents_[static_cast<std::size_t>(occupant)];
            const auto& memberships = topology_.memberships(dependency.position);
            if (!dependency.active || dependency.scheduled()
                || dependency.schedule_group != iid_value
                || dependency.position != movement_origin_[static_cast<std::size_t>(occupant)]
                || std::find(memberships.begin(), memberships.end(), iid_value) == memberships.end()) {
                blocked_[iid] = true;
                break;
            }
            rescue_member_[static_cast<std::size_t>(occupant)] = iid_value;
            members.push_back(occupant);
        }

        std::vector<CellId> destinations;
        destinations.reserve(members.size());
        for (const AgentId id : members) {
            if (blocked_[iid]) break;
            const Agent& agent = agents_[static_cast<std::size_t>(id)];
            const CellId next = next_for(agent);
            if (block_intersection(agent.position, next, false)
                || scheduled_reserved_[static_cast<std::size_t>(next)] != 0
                || std::find(destinations.begin(), destinations.end(), next) != destinations.end()) {
                blocked_[iid] = true;
                break;
            }
            const AgentId occupant = occupancy_[static_cast<std::size_t>(next)];
            if (occupant != kNoAgent && occupant != id
                && rescue_member_[static_cast<std::size_t>(occupant)] != iid_value) {
                blocked_[iid] = true;
                break;
            }
            if (occupant != kNoAgent && occupant != id
                && next_for(agents_[static_cast<std::size_t>(occupant)]) == agent.position) {
                blocked_[iid] = true;
                break;
            }
            destinations.push_back(next);
        }
        if (blocked_[iid]) {
            for (const AgentId id : members) rescue_member_[static_cast<std::size_t>(id)] = kNoGroup;
            continue;
        }
        rescue_group_[iid] = 1;
        for (const CellId destination : destinations)
            scheduled_reserved_[static_cast<std::size_t>(destination)] = 1;
    }

    for (const IntersectionId iid_value : deadlock_queue_) {
        const auto iid = static_cast<std::size_t>(iid_value);
        if (rescue_group_[iid] == 0) continue;
        for (const Agent& agent : agents_) {
            if (agent.active && rescue_member_[static_cast<std::size_t>(agent.id)] == iid_value)
                occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
        }
        for (Agent& agent : agents_) {
            const auto index = static_cast<std::size_t>(agent.id);
            if (!agent.active || rescue_member_[index] != iid_value) continue;
            const CellId forced = pibt_forced_next_[index];
            if (!agent.scheduled() && agent.wait_steps > 0 && stats_.waits > 0) --stats_.waits;
            if (agent.scheduled() || forced == kInvalidCell) {
                if (!agent.scheduled()) update_available_on_move(agent.position, agent.intended_cell());
                move_agent(agent);
            } else {
                update_available_on_move(agent.position, forced);
                move_agent_to(agent, forced);
            }
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        }
    }
    for (Agent& agent : agents_) {
        if (!agent.active || !agent.scheduled()) continue;
        const auto iid = static_cast<std::size_t>(agent.schedule_group);
        if (iid < rescue_group_.size() && rescue_group_[iid] != 0) continue;
        if (iid < blocked_.size() && blocked_[iid]) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::ScheduleGroup;
            continue;
        }
        occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
        move_agent(agent);
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
    }

    if (workload_ == WorkloadMode::Lifelong) {
        for (Agent& agent : agents_) {
            if (!agent.active || agent.awaiting_goal || agent.position != agent.goal) continue;
            agent.completed = true;
            agent.awaiting_goal = !despawn_at_goal_;
            if (despawn_at_goal_) {
                agent.active = false;
                agent.scheduling_remaining = 0;
                agent.schedule_group = kNoGroup;
            }
            ++agent.tasks_completed;
            ++stats_.completed_tasks;
            stats_.total_task_latency += stats_.timestep - agent.task_started_timestep;
        }
        if (!despawn_at_goal_) assign_lifelong_goals();
    } else {
        for (Agent& agent : agents_) {
            if (!agent.active) continue;
            const bool at_goal = agent.position == agent.goal;
            if (at_goal && !agent.completed) {
                agent.completed = true;
                ++stats_.completed;
                if (despawn_at_goal_) {
                    agent.active = false;
                    agent.scheduling_remaining = 0;
                    agent.schedule_group = kNoGroup;
                }
            } else if (!at_goal && agent.completed) {
                agent.completed = false;
                --stats_.completed;
            }
        }
    }
    return true;
}

void Simulator::assign_lifelong_goals() {
    std::fill(occupancy_.begin(), occupancy_.end(), kNoAgent);
    for (const Agent& agent : agents_) if (agent.active)
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;

    for (Agent& agent : agents_) {
        if (!agent.active || !agent.awaiting_goal || agent.scheduled()) continue;
        if (agent.schedule_group >= 0
            && deadlock_active_[static_cast<std::size_t>(agent.schedule_group)] != 0) continue;

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
        agent.scheduling_remaining = 0;
        agent.schedule_group = kNoGroup;
        agent.wait_steps = 0;
        agent.wait_reason = WaitReason::None;
        agent.completed = false;
        agent.awaiting_goal = false;
        agent.task_started_timestep = stats_.timestep;
    }
}

bool Simulator::recover_stalled_intersections(const std::vector<std::vector<AgentId>>& members,
                                              const std::vector<bool>& stalled) {
    const auto active = [&](const IntersectionId iid) {
        return deadlock_active_[static_cast<std::size_t>(iid)] != 0;
    };
    bool changed = false;
    const auto& intersections = topology_.intersections();
    for (const Intersection& source : intersections) {
        const std::size_t source_index = static_cast<std::size_t>(source.id);
        if (!stalled[source_index] || active(source.id)) continue;

        struct CycleCandidate {
            std::size_t direction{};
            std::vector<std::array<IntersectionId, 4>> cycles;
        };
        std::vector<CycleCandidate> candidates;
        for (std::size_t d = 0; d < 4; ++d) {
            const IntersectionId b = source.neighbors[d];
            if (b < 0 || stalled[static_cast<std::size_t>(b)] || active(b)) continue;
            const Intersection& bi = intersections[static_cast<std::size_t>(b)];
            std::vector<IntersectionId> around_b;
            for (const IntersectionId neighbor : bi.neighbors)
                if (neighbor >= 0 && neighbor != source.id) around_b.push_back(neighbor);
            CycleCandidate candidate;
            candidate.direction = d;
            for (std::size_t i = 0; i < around_b.size(); ++i) {
                for (std::size_t j = i + 1; j < around_b.size(); ++j) {
                    const IntersectionId c = around_b[i];
                    const IntersectionId e = around_b[j];
                    const Intersection& ci = intersections[static_cast<std::size_t>(c)];
                    const Intersection& ei = intersections[static_cast<std::size_t>(e)];
                    for (const IntersectionId common : ci.neighbors) {
                        if (common < 0 || common == b || common == c || common == e) continue;
                        if (std::find(ei.neighbors.begin(), ei.neighbors.end(), common) != ei.neighbors.end())
                            candidate.cycles.push_back({b, c, common, e});
                    }
                }
            }
            if (!candidate.cycles.empty()) candidates.push_back(std::move(candidate));
        }
        if (candidates.empty()) continue;
        std::uniform_int_distribution<std::size_t> choose_candidate(0, candidates.size() - 1);
        CycleCandidate& selected = candidates[choose_candidate(rng_)];
        std::uniform_int_distribution<std::size_t> choose_cycle(0, selected.cycles.size() - 1);
        std::array<IntersectionId, 4> cycle = selected.cycles[choose_cycle(rng_)];
        std::bernoulli_distribution reverse_cycle(0.5);
        if (reverse_cycle(rng_)) std::swap(cycle[1], cycle[3]);
        const std::size_t source_direction = selected.direction;

        const auto& escape_arm = source.arms[source_direction];
        for (const AgentId id : members[source_index]) {
            Agent& agent = agents_[static_cast<std::size_t>(id)];
            if (!agent.active || agent.scheduled()) continue;
            if (agent.position != source.center
                && std::find(escape_arm.begin(), escape_arm.end(), agent.position) == escape_arm.end()) continue;
            const CellId first_center = intersections[static_cast<std::size_t>(cycle[0])].center;
            if (std::find(agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor),
                          agent.route.end(), first_center) != agent.route.end()) continue;

            std::vector<CellId> detour{agent.position};
            CellId cursor = agent.position;
            bool valid = true;
            const std::array<CellId, 6> waypoints{{
                intersections[static_cast<std::size_t>(cycle[0])].center,
                intersections[static_cast<std::size_t>(cycle[1])].center,
                intersections[static_cast<std::size_t>(cycle[2])].center,
                intersections[static_cast<std::size_t>(cycle[3])].center,
                intersections[static_cast<std::size_t>(cycle[0])].center,
                agent.position}};
            for (const CellId waypoint : waypoints) {
                auto segment = planner_->plan(cursor, waypoint);
                if (segment.empty()) {
                    valid = false;
                    break;
                }
                detour.insert(detour.end(), segment.begin() + 1, segment.end());
                cursor = waypoint;
            }
            if (!valid) continue;
            std::vector<CellId> inserted;
            inserted.reserve(agent.route.size() + detour.size());
            inserted.insert(inserted.end(), agent.route.begin(),
                agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor + 1));
            inserted.insert(inserted.end(), detour.begin() + 1, detour.end());
            inserted.insert(inserted.end(),
                agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor + 1), agent.route.end());
            agent.route = std::move(inserted);
            changed = true;
        }
    }
    return changed;
}

std::vector<CellId> Simulator::plan_global(const CellId start, const CellId goal) {
    if (start == goal) return {start};
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
