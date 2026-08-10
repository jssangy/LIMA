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

Simulator::Simulator(GridMap map, const std::span<const Task> tasks, const PlannerKind planner_kind)
    : map_(std::move(map)), planner_(make_planner(planner_kind, map_)), topology_(IntersectionTopology::build(map_)),
      resolver_(map_) {
    agents_.reserve(tasks.size());
    std::unordered_set<CellId> occupied;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const CellId start = map_.in_bounds(tasks[i].start) ? map_.cell(tasks[i].start) : kInvalidCell;
        const CellId goal = map_.in_bounds(tasks[i].goal) ? map_.cell(tasks[i].goal) : kInvalidCell;
        if (!map_.traversable(start) || !map_.traversable(goal)) throw std::runtime_error("task endpoint is not traversable");
        if (!occupied.insert(start).second) throw std::runtime_error("duplicate task start position");
        auto route = plan_global(start, goal);
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
    intersection_available_.resize(topology_.intersections().size());
    for (const Intersection& intersection : topology_.intersections())
        intersection_available_[static_cast<std::size_t>(intersection.id)] = static_cast<int>(scheduling_capacity(intersection));
    for (const Agent& agent : agents_) if (agent.active) {
        for (const IntersectionId iid : topology_.memberships(agent.position))
            --intersection_available_[static_cast<std::size_t>(iid)];
    }
}

bool Simulator::step() {
    if (done()) return false;
    ++stats_.timestep;
    const std::uint64_t moves_before = stats_.committed_moves;

    std::vector<std::size_t> inside_counts(topology_.intersections().size(), 0);
    std::vector<std::vector<AgentId>> intersection_members(topology_.intersections().size());
    for (const Agent& agent : agents_) if (agent.active) {
        for (const IntersectionId iid : topology_.memberships(agent.position)) {
            ++inside_counts[static_cast<std::size_t>(iid)];
            intersection_members[static_cast<std::size_t>(iid)].push_back(agent.id);
        }
    }

    std::vector<IntersectionId> schedule_candidates;
    std::vector<bool> stalled_intersections(topology_.intersections().size(), false);
    for (const Intersection& intersection : topology_.intersections()) {
        const auto intents = collect_intents(intersection, agents_,
            intersection_members[static_cast<std::size_t>(intersection.id)]);
        if (intents.size() < 2) continue;
        const bool event_trigger = std::any_of(intents.begin(), intents.end(), [&](const IntersectionIntent& intent) {
            if (intent.position == intersection.center) return true;
            for (const auto& arm : intersection.arms) if (!arm.empty() && intent.position == arm.back()) return true;
            return false;
        });
        const bool stalled_trigger = std::all_of(intents.begin(), intents.end(), [&](const IntersectionIntent& intent) {
            return agents_[static_cast<std::size_t>(intent.agent)].wait_steps > 0;
        });
        stalled_intersections[static_cast<std::size_t>(intersection.id)] = stalled_trigger;
        if ((!event_trigger && !stalled_trigger) || !has_intersection_deadlock(intersection, intents)) continue;
        ++stats_.detected_deadlocks;
        schedule_candidates.push_back(intersection.id);
    }
    std::sort(schedule_candidates.begin(), schedule_candidates.end(), [&](const IntersectionId lhs, const IntersectionId rhs) {
        return inside_counts[static_cast<std::size_t>(lhs)] < inside_counts[static_cast<std::size_t>(rhs)];
    });
    for (const IntersectionId iid : schedule_candidates) {
        const Intersection& intersection = topology_.intersections()[static_cast<std::size_t>(iid)];
        const auto intents = collect_intents(intersection, agents_,
            intersection_members[static_cast<std::size_t>(iid)]);
        bool already_scheduled = false;
        for (const auto& intent : intents) {
            const Agent& agent = agents_[static_cast<std::size_t>(intent.agent)];
            already_scheduled = already_scheduled || agent.scheduled();
        }
        bool active_neighbor = false;
        for (const auto& [group, active_id] : active_schedule_intersections_) {
            (void)group;
            if (active_id == intersection.id || std::find(intersection.neighbors.begin(), intersection.neighbors.end(), active_id)
                    != intersection.neighbors.end()) {
                active_neighbor = true;
                break;
            }
        }
        std::array<int, 4> quotas{};
        int quota_sum = 0;
        for (std::size_t d = 0; d < 4; ++d) {
            const int arm_capacity = static_cast<int>(intersection.arms[d].size());
            if (arm_capacity == 0) continue;
            int initial_need = 0;
            for (const auto& intent : intents) if (static_cast<std::size_t>(intent.current) == d) ++initial_need;
            int neighbor_available = arm_capacity;
            const IntersectionId neighbor = intersection.neighbors[d];
            if (neighbor >= 0) {
                const auto& neighbor_intersection = topology_.intersections()[static_cast<std::size_t>(neighbor)];
                (void)neighbor_intersection;
                neighbor_available = std::max(0, intersection_available_[static_cast<std::size_t>(neighbor)]);
            }
            quotas[d] = std::min(arm_capacity, neighbor_available + initial_need);
            quota_sum += quotas[d];
        }
        const auto final_lengths = predict_final_stack_lengths(intersection, intents, quotas);
        std::array<int, 4> initial_lengths{};
        for (const auto& intent : intents) {
            const auto d = static_cast<std::size_t>(intent.current);
            if (d < 4) ++initial_lengths[d];
        }
        bool reserved = false;
        if (!already_scheduled && !active_neighbor
            && inside_counts[static_cast<std::size_t>(iid)] <= scheduling_capacity(intersection)
            && quota_sum >= static_cast<int>(intents.size())
            && final_lengths) {
            for (std::size_t d = 0; d < 4; ++d) {
                const IntersectionId neighbor = intersection.neighbors[d];
                if (neighbor < 0) continue;
                intersection_available_[static_cast<std::size_t>(neighbor)] -= (*final_lengths)[d] - initial_lengths[d];
            }
            reserved = true;
        }
        if (reserved && coordinator_.schedule(intersection, agents_, intents, quotas, next_schedule_group_)) {
            DischargeReservation reservation;
            reservation.source = intersection.id;
            for (std::size_t d = 0; d < 4; ++d)
                reservation.remaining[d] = std::max(0, (*final_lengths)[d] - initial_lengths[d]);
            discharge_reservations_[next_schedule_group_] = reservation;
            active_schedule_intersections_[next_schedule_group_] = intersection.id;
            ++next_schedule_group_;
        } else if (reserved) {
            for (std::size_t d = 0; d < 4; ++d) {
                const IntersectionId neighbor = intersection.neighbors[d];
                if (neighbor < 0) continue;
                intersection_available_[static_cast<std::size_t>(neighbor)] += (*final_lengths)[d] - initial_lengths[d];
            }
        }
    }

    recover_stalled_intersections(intersection_members, stalled_intersections);

    std::vector<MoveIntent> intents;
    std::vector<std::size_t> agent_indices;
    std::vector<WaitReason> forced_wait_reasons;
    std::vector<int> admission_remaining = intersection_available_;
    auto provisional_discharge = discharge_reservations_;
    std::vector<std::int32_t> discharge_claim_groups;
    std::vector<int> discharge_claim_directions;
    intents.reserve(agents_.size() - static_cast<std::size_t>(stats_.completed));
    for (std::size_t i = 0; i < agents_.size(); ++i) {
        const Agent& agent = agents_[i];
        if (!agent.active) continue;
        CellId intended = agent.intended_cell();
        WaitReason forced_wait = WaitReason::None;
        std::int32_t discharge_claim_group = kNoGroup;
        int discharge_claim_direction = -1;
        if (!agent.scheduled() && intended != agent.position) {
            std::vector<IntersectionId> admitted;
            for (const IntersectionId iid : topology_.memberships(intended)) {
                const bool entering = std::find(topology_.memberships(agent.position).begin(),
                                                topology_.memberships(agent.position).end(), iid)
                    == topology_.memberships(agent.position).end();
                const bool reserved = std::any_of(active_schedule_intersections_.begin(), active_schedule_intersections_.end(),
                    [&](const auto& entry) { return entry.second == iid; });
                int eligible_direction = -1;
                if (entering && agent.discharge_group != kNoGroup) {
                    const auto reservation_it = provisional_discharge.find(agent.discharge_group);
                    if (reservation_it != provisional_discharge.end()) {
                        const auto& reservation = reservation_it->second;
                        const auto& source_memberships = topology_.memberships(agent.position);
                        if (std::find(source_memberships.begin(), source_memberships.end(), reservation.source)
                            != source_memberships.end()) {
                            const auto& source = topology_.intersections()[static_cast<std::size_t>(reservation.source)];
                            for (std::size_t d = 0; d < 4; ++d) {
                                if (source.neighbors[d] == iid && reservation.remaining[d] > 0) {
                                    eligible_direction = static_cast<int>(d);
                                    break;
                                }
                            }
                        }
                    }
                }
                if ((reserved && entering)
                    || (entering && eligible_direction < 0
                        && admission_remaining[static_cast<std::size_t>(iid)] <= 0)) {
                    intended = agent.position;
                    forced_wait = reserved ? WaitReason::IntersectionReserved : WaitReason::IntersectionCapacity;
                    for (const IntersectionId admitted_iid : admitted)
                        ++admission_remaining[static_cast<std::size_t>(admitted_iid)];
                    if (discharge_claim_group != kNoGroup) {
                        ++provisional_discharge[discharge_claim_group].remaining[
                            static_cast<std::size_t>(discharge_claim_direction)];
                        discharge_claim_group = kNoGroup;
                        discharge_claim_direction = -1;
                    }
                    break;
                }
                if (entering) {
                    if (eligible_direction >= 0) {
                        discharge_claim_group = agent.discharge_group;
                        discharge_claim_direction = eligible_direction;
                        --provisional_discharge[discharge_claim_group].remaining[
                            static_cast<std::size_t>(eligible_direction)];
                    } else {
                        --admission_remaining[static_cast<std::size_t>(iid)];
                        admitted.push_back(iid);
                    }
                }
            }
        }
        intents.push_back({agent.id, agent.position, intended, agent.schedule_group, agent.wait_steps});
        agent_indices.push_back(i);
        forced_wait_reasons.push_back(forced_wait);
        discharge_claim_groups.push_back(discharge_claim_group);
        discharge_claim_directions.push_back(discharge_claim_direction);
    }

    const MoveResolution resolution = resolver_.resolve(intents);
    std::vector<CellId> next_positions;
    next_positions.reserve(intents.size());
    for (std::size_t i = 0; i < intents.size(); ++i) next_positions.push_back(resolution.approved[i] ? intents[i].to : intents[i].from);

    // Atomic commit: no Agent is mutated before every next position has been resolved and validated.
    for (std::size_t i = 0; i < intents.size(); ++i) {
        Agent& agent = agents_[agent_indices[i]];
        if (forced_wait_reasons[i] != WaitReason::None) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = forced_wait_reasons[i];
            continue;
        }
        if (resolution.approved[i]) {
            agent.position = next_positions[i];
            if (intents[i].from != intents[i].to) {
                const auto& from_memberships = topology_.memberships(intents[i].from);
                const auto& to_memberships = topology_.memberships(intents[i].to);
                for (const IntersectionId iid : from_memberships) {
                    if (std::find(to_memberships.begin(), to_memberships.end(), iid) == to_memberships.end()) {
                        const int capacity = static_cast<int>(scheduling_capacity(
                            topology_.intersections()[static_cast<std::size_t>(iid)]));
                        intersection_available_[static_cast<std::size_t>(iid)] =
                            std::min(capacity, intersection_available_[static_cast<std::size_t>(iid)] + 1);
                    }
                }
                for (const IntersectionId iid : to_memberships) {
                    if (std::find(from_memberships.begin(), from_memberships.end(), iid) == from_memberships.end()) {
                        bool consumed_reservation = false;
                        if (discharge_claim_groups[i] != kNoGroup && discharge_claim_directions[i] >= 0) {
                            auto reservation_it = discharge_reservations_.find(discharge_claim_groups[i]);
                            if (reservation_it != discharge_reservations_.end()) {
                                const std::size_t d = static_cast<std::size_t>(discharge_claim_directions[i]);
                                const auto& source = topology_.intersections()[
                                    static_cast<std::size_t>(reservation_it->second.source)];
                                if (source.neighbors[d] == iid && reservation_it->second.remaining[d] > 0) {
                                    --reservation_it->second.remaining[d];
                                    consumed_reservation = true;
                                }
                            }
                        }
                        if (!consumed_reservation)
                            intersection_available_[static_cast<std::size_t>(iid)] =
                                std::max(0, intersection_available_[static_cast<std::size_t>(iid)] - 1);
                    }
                }
                if (agent.discharge_group != kNoGroup) {
                    const auto reservation_it = discharge_reservations_.find(agent.discharge_group);
                    if (reservation_it == discharge_reservations_.end()
                        || std::find(to_memberships.begin(), to_memberships.end(), reservation_it->second.source)
                            == to_memberships.end()) {
                        agent.discharge_group = kNoGroup;
                    }
                }
            }
            if (agent.scheduled()) {
                if (agent.schedule_cursor + 1 < agent.schedule_route.size()) ++agent.schedule_cursor;
            } else if (agent.route_cursor + 1 < agent.route.size()) {
                ++agent.route_cursor;
            }
            if (intents[i].from != intents[i].to) {
                ++agent.moves;
                ++stats_.committed_moves;
            }
            agent.wait_steps = 0;
            agent.wait_reason = agent.scheduled() && intents[i].from == intents[i].to
                ? WaitReason::ScheduledHold : WaitReason::None;
        } else {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = resolution.reasons[i];
        }
    }

    for (Agent& agent : agents_) {
        if (!agent.active || !agent.scheduled() || agent.schedule_cursor + 1 < agent.schedule_route.size()) continue;
        const auto active = active_schedule_intersections_.find(agent.schedule_group);
        if (active != active_schedule_intersections_.end()) reconnect_to_original_route(agent, active->second);
        else agent.route = plan_global(agent.position, agent.goal);
        agent.discharge_group = agent.schedule_group;
        agent.schedule_route.clear();
        agent.schedule_cursor = 0;
        agent.schedule_group = kNoGroup;
        if (agent.route.empty()) throw std::runtime_error("failed to reconnect scheduled route to global goal");
    }
    for (auto it = active_schedule_intersections_.begin(); it != active_schedule_intersections_.end();) {
        const std::int32_t group = it->first;
        const bool still_active = std::any_of(agents_.begin(), agents_.end(),
            [&](const Agent& agent) { return agent.active && agent.schedule_group == group; });
        if (!still_active) it = active_schedule_intersections_.erase(it);
        else ++it;
    }

    for (Agent& agent : agents_) {
        if (agent.active && agent.position == agent.goal) {
            agent.active = false;
            agent.discharge_group = kNoGroup;
            ++stats_.completed;
        }
    }
    for (auto it = discharge_reservations_.begin(); it != discharge_reservations_.end();) {
        const std::int32_t group = it->first;
        const bool schedule_active = active_schedule_intersections_.contains(group);
        const bool has_owner = std::any_of(agents_.begin(), agents_.end(), [&](const Agent& agent) {
            return agent.active && agent.discharge_group == group;
        });
        if (schedule_active || has_owner) {
            ++it;
            continue;
        }
        const auto& source = topology_.intersections()[static_cast<std::size_t>(it->second.source)];
        for (std::size_t d = 0; d < 4; ++d) {
            const IntersectionId neighbor = source.neighbors[d];
            if (neighbor >= 0 && it->second.remaining[d] > 0) {
                const int capacity = static_cast<int>(scheduling_capacity(
                    topology_.intersections()[static_cast<std::size_t>(neighbor)]));
                intersection_available_[static_cast<std::size_t>(neighbor)] = std::min(
                    capacity, intersection_available_[static_cast<std::size_t>(neighbor)] + it->second.remaining[d]);
            }
        }
        it = discharge_reservations_.erase(it);
    }
    if (stats_.committed_moves == moves_before && !done()) {
        ++stalled_timesteps_;
        if (stalled_timesteps_ >= 4 && recover_from_stall()) stalled_timesteps_ = 0;
    } else {
        stalled_timesteps_ = 0;
    }
    return true;
}

bool Simulator::recover_from_stall() {
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < agents_.size(); ++i) {
        const Agent& agent = agents_[i];
        if (agent.active && !agent.scheduled() && agent.intended_cell() != agent.position) candidates.push_back(i);
    }
    std::sort(candidates.begin(), candidates.end(), [&](const std::size_t lhs, const std::size_t rhs) {
        if (agents_[lhs].wait_steps != agents_[rhs].wait_steps) return agents_[lhs].wait_steps > agents_[rhs].wait_steps;
        return agents_[lhs].id < agents_[rhs].id;
    });

    std::vector<std::uint8_t> base_occupied(static_cast<std::size_t>(map_.cell_count()), 0);
    for (const Agent& agent : agents_) if (agent.active) base_occupied[static_cast<std::size_t>(agent.position)] = 1;
    for (const std::size_t index : candidates) {
        Agent& agent = agents_[index];
        auto blocked = base_occupied;
        blocked[static_cast<std::size_t>(agent.position)] = 0;
        blocked[static_cast<std::size_t>(agent.goal)] = 0;
        auto detour = plan_avoiding(map_, agent.position, agent.goal, blocked);
        if (detour.size() < 2) {
            std::fill(blocked.begin(), blocked.end(), 0);
            const CellId blocked_next = agent.intended_cell();
            if (blocked_next != agent.goal) blocked[static_cast<std::size_t>(blocked_next)] = 1;
            detour = plan_avoiding(map_, agent.position, agent.goal, blocked);
        }
        if (detour.size() >= 2 && detour[1] != agent.intended_cell()) {
            agent.route = std::move(detour);
            agent.route_cursor = 0;
            agent.wait_steps = 0;
            return true;
        }
    }
    return false;
}

bool Simulator::recover_stalled_intersections(const std::vector<std::vector<AgentId>>& members,
                                              const std::vector<bool>& stalled) {
    AStarPlanner detour_planner(map_);
    const auto active = [&](const IntersectionId iid) {
        return std::any_of(active_schedule_intersections_.begin(), active_schedule_intersections_.end(),
            [&](const auto& entry) { return entry.second == iid; });
    };
    bool changed = false;
    const auto& intersections = topology_.intersections();
    for (const Intersection& source : intersections) {
        const std::size_t source_index = static_cast<std::size_t>(source.id);
        if (!stalled[source_index] || active(source.id)) continue;

        std::size_t source_direction = 4;
        std::array<IntersectionId, 4> cycle{{-1, -1, -1, -1}};
        for (std::size_t d = 0; d < 4 && cycle[0] < 0; ++d) {
            const IntersectionId b = source.neighbors[d];
            if (b < 0 || stalled[static_cast<std::size_t>(b)] || active(b)) continue;
            const Intersection& bi = intersections[static_cast<std::size_t>(b)];
            for (const IntersectionId c : bi.neighbors) {
                if (c < 0 || c == source.id) continue;
                for (const IntersectionId e : bi.neighbors) {
                    if (e < 0 || e == source.id || e == c) continue;
                    const Intersection& ci = intersections[static_cast<std::size_t>(c)];
                    for (const IntersectionId common : ci.neighbors) {
                        if (common < 0 || common == b || common == c || common == e) continue;
                        const Intersection& ei = intersections[static_cast<std::size_t>(e)];
                        if (std::find(ei.neighbors.begin(), ei.neighbors.end(), common) == ei.neighbors.end()) continue;
                        source_direction = d;
                        cycle = {b, c, common, e};
                        break;
                    }
                    if (cycle[0] >= 0) break;
                }
                if (cycle[0] >= 0) break;
            }
        }
        if (cycle[0] < 0 || source_direction >= 4) continue;

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
                auto segment = detour_planner.plan(cursor, waypoint);
                if (segment.empty()) {
                    valid = false;
                    break;
                }
                detour.insert(detour.end(), segment.begin() + 1, segment.end());
                cursor = waypoint;
            }
            if (!valid) continue;
            // Drop any older recovery loops instead of recursively appending them.
            // The loop returns to the current cell, so a fresh global suffix is
            // equivalent and keeps route memory bounded under heavy congestion.
            auto continuation = plan_global(agent.position, agent.goal);
            if (continuation.size() > 1) detour.insert(detour.end(), continuation.begin() + 1, continuation.end());
            agent.route = std::move(detour);
            agent.route_cursor = 0;
            agent.wait_steps = 0;
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
        const int middle = (a + b) / 2;
        std::sort(result.begin(), result.end(), [&](const int lhs, const int rhs) {
            return std::abs(lhs - middle) < std::abs(rhs - middle);
        });
        if (result.size() > 16) result.resize(16);
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
        if (x_align) for (const int y : candidates(center_ys, source.y, destination.y)) {
            auto route = via({{*x_align, source.y}, {*x_align, y}, {destination.x, y}, destination});
            if (!route.empty()) return route;
        }
    } else if (horizontal_goal) {
        const auto y_align = nearest(center_ys, source.y, destination.y, source.y);
        if (y_align) for (const int x : candidates(center_xs, source.x, destination.x)) {
            auto route = via({{source.x, *y_align}, {x, *y_align}, {x, destination.y}, destination});
            if (!route.empty()) return route;
        }
    }
    return planner_->plan(start, goal);
}

void Simulator::reconnect_to_original_route(Agent& agent, const IntersectionId intersection_id) {
    for (std::size_t i = agent.route_cursor; i < agent.route.size(); ++i) {
        if (agent.route[i] == agent.position) {
            agent.route_cursor = i;
            return;
        }
    }

    const Intersection& intersection = topology_.intersections()[static_cast<std::size_t>(intersection_id)];
    std::size_t rejoin = agent.route.size();
    bool reached_center = false;
    for (std::size_t i = agent.route_cursor; i < agent.route.size(); ++i) {
        if (agent.route[i] == intersection.center) reached_center = true;
        if (!reached_center) continue;
        const auto& memberships = topology_.memberships(agent.route[i]);
        const bool inside = std::find(memberships.begin(), memberships.end(), intersection_id) != memberships.end();
        if (inside) rejoin = i;
        else if (rejoin != agent.route.size()) break;
    }
    if (rejoin == agent.route.size()) {
        agent.route = plan_global(agent.position, agent.goal);
        agent.route_cursor = 0;
        return;
    }

    auto bridge = planner_->plan(agent.position, agent.route[rejoin]);
    if (bridge.empty()) {
        agent.route = plan_global(agent.position, agent.goal);
        agent.route_cursor = 0;
        return;
    }
    std::vector<CellId> connected = std::move(bridge);
    connected.insert(connected.end(), agent.route.begin() + static_cast<std::ptrdiff_t>(rejoin + 1), agent.route.end());
    agent.route = std::move(connected);
    agent.route_cursor = 0;
}

}  // namespace lima
