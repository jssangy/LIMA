#include "lima/intersection/gating.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace lima {

int scheduling_capacity(const Intersection& intersection, const IsolationConfig& config) {
    std::size_t total = 0;
    std::size_t longest = 0;
    for (const auto& arm : intersection.arms) {
        total += arm.size();
        longest = std::max(longest, arm.size());
    }
    int capacity = static_cast<int>(total - longest);
    if (config.formula == CapacityFormula::SumMinusMaxPlusOne) ++capacity;
    capacity = std::max(0, capacity - config.margin);
    if (config.cap >= 0) capacity = std::min(capacity, config.cap);
    return capacity;
}

std::vector<RecirculationDischarge::Event> RecirculationDischarge::run(const Context& context) const {
    const auto active = [&](const IntersectionId iid) {
        return context.deadlock_active[static_cast<std::size_t>(iid)] != 0;
    };
    std::vector<Event> events;
    const auto& intersections = context.topology.intersections();
    for (const Intersection& source : intersections) {
        const std::size_t source_index = static_cast<std::size_t>(source.id);
        if (!context.stalled[source_index] || active(source.id)) continue;

        struct CycleCandidate {
            std::size_t direction{};
            std::vector<std::array<IntersectionId, 4>> cycles;
        };
        std::vector<CycleCandidate> candidates;
        for (std::size_t d = 0; d < 4; ++d) {
            const IntersectionId b = source.neighbors[d];
            if (b < 0 || active(b)) continue;
            if (context.stalled[static_cast<std::size_t>(b)] && !config_.allow_stalled_neighbor) continue;
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
        std::size_t candidate_index = 0;
        std::size_t cycle_index = 0;
        bool reversed = false;
        if (config_.avail_weighted && context.available) {
            // Escape toward the neighbor advertising the most admission slack;
            // uses only the 1-hop availability the modules already exchange.
            int best_slack = std::numeric_limits<int>::min();
            for (std::size_t ci = 0; ci < candidates.size(); ++ci) {
                const IntersectionId b = source.neighbors[candidates[ci].direction];
                const int slack = (*context.available)[static_cast<std::size_t>(b)];
                if (slack > best_slack) {
                    best_slack = slack;
                    candidate_index = ci;
                }
            }
            if (config_.deterministic_cycle && context.stale_loads) {
                // Composite: within the slack-chosen direction, prefer the
                // recirculation loop over the least-loaded intersections using
                // the one-cycle-stale relayed counts.
                std::size_t best_load = std::numeric_limits<std::size_t>::max();
                const auto& cycles = candidates[candidate_index].cycles;
                for (std::size_t ki = 0; ki < cycles.size(); ++ki) {
                    std::size_t load = 0;
                    for (const IntersectionId node : cycles[ki])
                        load += (*context.stale_loads)[static_cast<std::size_t>(node)];
                    if (load < best_load) {
                        best_load = load;
                        cycle_index = ki;
                    }
                }
            }
        } else if (config_.deterministic_cycle) {
            // Least-loaded choice: prefer the cycle whose member intersections
            // currently hold the fewest agents, breaking ties by index.
            std::size_t best_load = std::numeric_limits<std::size_t>::max();
            for (std::size_t ci = 0; ci < candidates.size(); ++ci) {
                for (std::size_t ki = 0; ki < candidates[ci].cycles.size(); ++ki) {
                    std::size_t load = 0;
                    for (const IntersectionId node : candidates[ci].cycles[ki])
                        load += context.members[static_cast<std::size_t>(node)].size();
                    if (load < best_load) {
                        best_load = load;
                        candidate_index = ci;
                        cycle_index = ki;
                    }
                }
            }
        } else {
            std::uniform_int_distribution<std::size_t> choose_candidate(0, candidates.size() - 1);
            candidate_index = choose_candidate(context.rng);
            std::uniform_int_distribution<std::size_t> choose_cycle(0, candidates[candidate_index].cycles.size() - 1);
            cycle_index = choose_cycle(context.rng);
            std::bernoulli_distribution reverse_cycle(0.5);
            reversed = reverse_cycle(context.rng);
        }
        CycleCandidate& selected = candidates[candidate_index];
        std::array<IntersectionId, 4> cycle = selected.cycles[cycle_index];
        if (reversed) std::swap(cycle[1], cycle[3]);
        const std::size_t source_direction = selected.direction;

        Event event;
        event.intersection = source.id;
        const auto& escape_arm = source.arms[source_direction];
        for (const AgentId id : context.members[source_index]) {
            Agent& agent = context.agents[static_cast<std::size_t>(id)];
            if (!agent.active || agent.scheduled()) continue;
            if (!config_.all_arms && agent.position != source.center
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
                auto segment = context.planner.plan(cursor, waypoint);
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
            ++event.rerouted;
            event.loop_cells = std::max(event.loop_cells, detour.size());
            event.agent_ids.push_back(agent.id);
        }
        if (event.rerouted > 0) events.push_back(event);
    }
    return events;
}

}  // namespace lima
