#include "lima/simulation/simulator.hpp"

#include "lima/intersection/deadlock_detector.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace lima {
namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

int cell_distance(const GridMap& map, const CellId first, const CellId second) {
    const Coord a = map.coord(first);
    const Coord b = map.coord(second);
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

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
      shuffle_rng_(config_.shuffle_order >= 0
          ? static_cast<std::uint64_t>(config_.shuffle_order) ^ 0xd1b54a32d192ed03ULL : 0ULL),
      failure_seed_(seed ^ 0xa0761d6478bd642fULL),
      planner_(make_planner(planner_kind, map_, rng_)),
      topology_(IntersectionTopology::build(map_)),
      solver_(make_solver(config_.solver)), coordinator_(*solver_, config_.pibt_corridor),
      discharge_(config_.discharge) {
    admission_seed_ = seed ^ 0xe7037ed1a0b428dbULL;
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
        agent.reference_route = agent.route;
        agents_.push_back(std::move(agent));
    }
    for (Agent& agent : agents_) {
        if (agent.position != agent.goal) continue;
        if (config_.goal_behavior == GoalBehavior::Disappear) {
            agent.active = false;
            ++stats_.completed;
        } else if (config_.goal_behavior == GoalBehavior::Stay) {
            agent.reached = true;
            ++stats_.completed;
        } else {
            agent.awaiting_goal = true;
            ++agent.tasks_completed;
            ++stats_.completed;
        }
    }
    initial_route_lengths_.reserve(agents_.size());
    for (const Agent& agent : agents_) initial_route_lengths_.push_back(agent.route.size());
    completion_steps_.resize(agents_.size(), 0);
    task_start_steps_.resize(agents_.size(), 0);
    execution_failed_.resize(agents_.size(), 0);
    if (!config_.metrics_dir.empty())
        metrics_ = std::make_unique<MetricsCollector>(config_.metrics_dir, map_, agents_);
    if (!config_.trace_path.empty()) {
        tracer_ = std::make_unique<StepTracer>(config_.trace_path, map_, config_.map_file, seed, agents_);
    }
    intersection_available_.resize(topology_.intersections().size());
    intersection_capacity_.resize(topology_.intersections().size());
    admission_reserve_.resize(topology_.intersections().size(), config_.isolation.hysteresis);
    admission_arm_mask_.resize(topology_.intersections().size(), 0x0fU);
    admission_requests_.resize(topology_.intersections().size());
    admission_arm_requests_.resize(topology_.intersections().size());
    admission_arm_max_wait_.resize(topology_.intersections().size());
    admission_deficit_.resize(topology_.intersections().size());
    admission_rr_cursor_.resize(topology_.intersections().size());
    admission_congestion_age_.resize(topology_.intersections().size());
    admission_window_.resize(topology_.intersections().size());
    admission_occupancy_ewma_.resize(topology_.intersections().size());
    admission_integral_.resize(topology_.intersections().size());
    admission_tokens_.resize(topology_.intersections().size());
    admission_probability_.resize(topology_.intersections().size());
    admission_previous_signal_.resize(topology_.intersections().size());
    admission_arm_probability_.resize(topology_.intersections().size());
    admission_arm_counter_.resize(topology_.intersections().size());
    scheduled_members_.resize(topology_.intersections().size());
    deadlock_waiting_.resize(topology_.intersections().size(), false);
    deadlock_active_.resize(topology_.intersections().size(), 0);
    deadlock_priority_.resize(topology_.intersections().size(), 0);
    for (const Intersection& intersection : topology_.intersections()) {
        const auto iid = static_cast<std::size_t>(intersection.id);
        intersection_capacity_[iid] = scheduling_capacity(intersection, config_.isolation);
        intersection_available_[iid] = intersection_capacity_[iid];
        admission_window_[iid] = static_cast<double>(intersection_capacity_[iid]);
        admission_tokens_[iid] = static_cast<double>(intersection_capacity_[iid]);
    }
    aimd_admission_.reset(
        agents_.size(), intersection_capacity_,
        {.multiplicative_decrease = config_.admission.parameter > 0.0
             ? config_.admission.parameter : 0.5,
         .additive_increase = config_.admission.secondary > 0.0
             ? config_.admission.secondary : 0.25});
    aimd_requests_.reserve(agents_.size());
    aimd_observations_.resize(topology_.intersections().size());
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
        goal_allocator_ = std::make_unique<GoalAllocator>(
            map_, agents_, seed ^ task_seed_salt, config_.lifelong_goal_sequences);
    }
    if (config_.pibt_corridor) {
        pibt_ = std::make_unique<PibtResolver>(map_, seed, config_.pibt_age_rate,
                                               config_.pibt_arm_retreat_last);
        pibt_eligible_.resize(agents_.size());
        pibt_priority_class_.resize(agents_.size());
        pibt_forced_next_.resize(agents_.size(), kInvalidCell);
        pibt_next_.resize(agents_.size(), kInvalidCell);
        rescue_candidate_.resize(intersection_count);
        rescue_group_.resize(intersection_count);
        rescue_member_.resize(agents_.size(), kNoGroup);
        scheduled_reserved_.resize(static_cast<std::size_t>(map_.cell_count()));
        movement_origin_.resize(agents_.size(), kInvalidCell);
    }
    if (goal_allocator_ && std::any_of(agents_.begin(), agents_.end(), [](const Agent& agent) {
            return agent.awaiting_goal;
        })) {
        for (const Agent& agent : agents_) if (agent.active)
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        assign_lifelong_goals();
    }
}

bool Simulator::sample_command_failure(const AgentId agent) {
    if (config_.failure_probability <= 0.0) return false;
    const auto index = static_cast<std::size_t>(agent);
    if (execution_failed_[index] != 0) return true;
    std::uint64_t counter = failure_seed_;
    counter ^= (static_cast<std::uint64_t>(agent) + 1ULL) * 0xd2b74407b1ce6e93ULL;
    counter ^= (stats_.timestep + 1ULL) * 0xca5a826395121157ULL;
    constexpr std::uint64_t scale = 1ULL << 53U;
    const auto threshold = static_cast<std::uint64_t>(
        config_.failure_probability * static_cast<double>(scale));
    const std::uint64_t sample = splitmix64(counter) >> 11U;
    if (sample >= threshold) return false;
    execution_failed_[index] = 1;
    ++stats_.command_failures;
    return true;
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

IntersectionId Simulator::entering_intersection(const CellId current, const CellId next) const {
    const auto& current_memberships = topology_.memberships(current);
    if (!current_memberships.empty() && !topology_.is_center(current)) return -1;
    if (!topology_.is_arm_tip(next)) return -1;
    for (const IntersectionId iid : topology_.memberships(next)) {
        if (std::find(current_memberships.begin(), current_memberships.end(), iid)
            == current_memberships.end()) return iid;
    }
    return -1;
}

void Simulator::update_admission_policy() {
    std::fill(admission_requests_.begin(), admission_requests_.end(), 0);
    std::fill(admission_arm_requests_.begin(), admission_arm_requests_.end(),
              std::array<std::size_t, 4>{});
    std::fill(admission_arm_max_wait_.begin(), admission_arm_max_wait_.end(),
              std::array<std::uint32_t, 4>{});
    aimd_requests_.clear();
    std::fill(admission_arm_mask_.begin(), admission_arm_mask_.end(), 0x0fU);

    for (const Agent& agent : agents_) if (agent.active && !agent.scheduled()) {
        const CellId next = agent.intended_cell();
        const IntersectionId entering = entering_intersection(agent.position, next);
        if (entering < 0) continue;
        const auto iid = static_cast<std::size_t>(entering);
        const Direction direction = topology_.intersections()[iid].direction_of(next);
        const auto d = static_cast<std::size_t>(direction);
        if (d >= 4) continue;
        ++admission_requests_[iid];
        ++admission_arm_requests_[iid][d];
        admission_arm_max_wait_[iid][d] = std::max(admission_arm_max_wait_[iid][d], agent.wait_steps);
        aimd_requests_.push_back({agent.id, entering, agent.wait_steps});
    }

    const auto policy = config_.admission.policy;
    if (policy == AdmissionPolicy::Aimd) {
        for (std::size_t iid = 0; iid < topology_.intersections().size(); ++iid) {
            const bool execution_delayed = std::any_of(
                members_[iid].begin(), members_[iid].end(), [&](const AgentId id) {
                    return agents_[static_cast<std::size_t>(id)].wait_reason
                        == WaitReason::ExecutionFailure;
                });
            aimd_observations_[iid] = {
                intersection_capacity_[iid], static_cast<int>(inside_counts_[iid]),
                stalled_[iid], execution_delayed};
        }
        aimd_admission_.update(aimd_observations_, aimd_requests_);
    }
    for (std::size_t iid = 0; iid < topology_.intersections().size(); ++iid) {
        const int capacity = std::max(1, intersection_capacity_[iid]);
        const int occupied = static_cast<int>(inside_counts_[iid]);
        const int available = std::max(0, capacity - occupied);
        const double ratio = static_cast<double>(occupied) / static_cast<double>(capacity);
        admission_occupancy_ewma_[iid] += 0.125 * (ratio - admission_occupancy_ewma_[iid]);
        int reserve = std::max(0, config_.isolation.hysteresis);

        switch (policy) {
        case AdmissionPolicy::Static:
            break;
        case AdmissionPolicy::FractionalReserve: {
            const double fraction = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.25;
            reserve = std::max(reserve, static_cast<int>(std::ceil(fraction * capacity)));
            break;
        }
        case AdmissionPolicy::RequestProportional: {
            const double gain = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 1.0;
            const int window = std::clamp(static_cast<int>(std::ceil(
                gain * static_cast<double>(admission_requests_[iid]))), 0, capacity);
            reserve = std::max(reserve, capacity - window);
            break;
        }
        case AdmissionPolicy::Backpressure: {
            const double gain = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 1.0;
            if (gain * static_cast<double>(admission_requests_[iid]) <= occupied)
                reserve = std::max(reserve, available);
            break;
        }
        case AdmissionPolicy::NeighborPressure: {
            double neighbor_ratio = 0.0;
            int neighbors = 0;
            for (const IntersectionId neighbor : topology_.intersections()[iid].neighbors) {
                if (neighbor < 0) continue;
                const auto n = static_cast<std::size_t>(neighbor);
                neighbor_ratio += static_cast<double>(prev_inside_counts_[n])
                    / static_cast<double>(std::max(1, intersection_capacity_[n]));
                ++neighbors;
            }
            if (neighbors > 0) neighbor_ratio /= static_cast<double>(neighbors);
            const double gain = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 1.0;
            const double request_pressure = gain * static_cast<double>(admission_requests_[iid])
                / static_cast<double>(capacity);
            if (request_pressure <= std::max(ratio, neighbor_ratio))
                reserve = std::max(reserve, available);
            break;
        }
        case AdmissionPolicy::Aimd: {
            // The reference controller owns acknowledged in-flight grants in
            // AcknowledgedAimdAdmission. Physical capacity remains governed
            // by intersection_available_ at movement commit time.
            break;
        }
        case AdmissionPolicy::Red:
            // The EWMA and deterministic per-request RED mark are evaluated
            // in admission_policy_blocks(); the ordinary capacity gate remains active.
            break;
        case AdmissionPolicy::Blue: {
            const double increase = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.02;
            const double decrease = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 0.002;
            const double maximum = config_.admission.tertiary > 0.0
                ? config_.admission.tertiary : 0.95;
            if (stalled_[iid] || occupied >= capacity)
                admission_probability_[iid] = std::min(
                    maximum, admission_probability_[iid] + increase);
            else if (occupied == 0 || (admission_requests_[iid] == 0 && ratio < 0.25))
                admission_probability_[iid] = std::max(
                    0.0, admission_probability_[iid] - decrease);
            break;
        }
        case AdmissionPolicy::Rem: {
            const double target = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.75;
            const double gain = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 0.05;
            const double scale = config_.admission.tertiary > 0.0
                ? config_.admission.tertiary : 1.0;
            const double offered = static_cast<double>(admission_requests_[iid])
                / static_cast<double>(capacity);
            admission_integral_[iid] = std::clamp(
                admission_integral_[iid] + gain * (ratio + offered - target), 0.0, 20.0);
            admission_probability_[iid] = 1.0 - std::exp(-scale * admission_integral_[iid]);
            break;
        }
        case AdmissionPolicy::Avq: {
            const double target = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.85;
            const double gain = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 0.10;
            admission_window_[iid] = std::clamp(
                admission_window_[iid] + gain * (target - ratio) * capacity,
                1.0, static_cast<double>(capacity));
            reserve = std::max(reserve, capacity
                - static_cast<int>(std::floor(admission_window_[iid])));
            break;
        }
        case AdmissionPolicy::Codel: {
            const std::uint32_t target = static_cast<std::uint32_t>(
                config_.admission.parameter > 0.0 ? config_.admission.parameter : 3.0);
            const std::uint32_t interval = static_cast<std::uint32_t>(
                config_.admission.secondary > 0.0 ? config_.admission.secondary : 5.0);
            std::uint32_t oldest = 0;
            for (const auto wait : admission_arm_max_wait_[iid]) oldest = std::max(oldest, wait);
            if (oldest > target && occupied > 0) ++admission_congestion_age_[iid];
            else admission_congestion_age_[iid] = 0;
            reserve = std::max(reserve, std::min(capacity - 1,
                static_cast<int>(admission_congestion_age_[iid] / std::max(1U, interval))));
            break;
        }
        case AdmissionPolicy::Pi: {
            const double target = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.75;
            const double kp = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 0.5;
            const double ki = config_.admission.tertiary > 0.0
                ? config_.admission.tertiary : 0.05;
            const double error = admission_occupancy_ewma_[iid] - target;
            admission_integral_[iid] = std::clamp(admission_integral_[iid] + error, -10.0, 10.0);
            const double control = std::clamp(kp * error + ki * admission_integral_[iid], 0.0, 0.9);
            reserve = std::max(reserve, static_cast<int>(std::ceil(control * capacity)));
            break;
        }
        case AdmissionPolicy::Pie: {
            const double target = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 5.0;
            const double alpha = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 0.02;
            const double beta = config_.admission.tertiary > 0.0
                ? config_.admission.tertiary : 0.002;
            double delay = 0.0;
            for (const auto wait : admission_arm_max_wait_[iid])
                delay = std::max(delay, static_cast<double>(wait));
            const double normalized_error = (delay - target) / std::max(1.0, target);
            const double normalized_delta = (delay - admission_previous_signal_[iid])
                / std::max(1.0, target);
            admission_probability_[iid] = std::clamp(
                admission_probability_[iid] + alpha * normalized_error
                    + beta * normalized_delta,
                0.0, 1.0);
            if (delay <= 0.5 * target)
                admission_probability_[iid] *= 0.98;
            admission_previous_signal_[iid] = delay;
            break;
        }
        case AdmissionPolicy::TokenBucket: {
            const double rate = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 1.0;
            const double bucket = config_.admission.secondary > 0.0
                ? config_.admission.secondary : static_cast<double>(capacity);
            admission_tokens_[iid] = std::min(bucket, admission_tokens_[iid] + rate);
            break;
        }
        case AdmissionPolicy::Sotl: {
            const double threshold = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 8.0;
            const std::uint32_t minimum_green = static_cast<std::uint32_t>(
                config_.admission.secondary > 0.0 ? config_.admission.secondary : 2.0);
            std::size_t selected = admission_rr_cursor_[iid] % 4U;
            ++admission_congestion_age_[iid];
            for (std::size_t d = 0; d < 4; ++d)
                if (d != selected) admission_arm_counter_[iid][d]
                    += static_cast<double>(admission_arm_requests_[iid][d]);
            std::size_t challenger = selected;
            for (std::size_t d = 0; d < 4; ++d) {
                if (admission_arm_requests_[iid][d] == 0) continue;
                if (admission_arm_counter_[iid][d] > admission_arm_counter_[iid][challenger])
                    challenger = d;
            }
            const bool current_empty = admission_arm_requests_[iid][selected] == 0;
            if (challenger != selected && (current_empty
                || (admission_congestion_age_[iid] >= minimum_green
                    && admission_arm_counter_[iid][challenger] >= threshold))) {
                selected = challenger;
                admission_rr_cursor_[iid] = static_cast<std::uint32_t>(selected);
                admission_congestion_age_[iid] = 0;
                admission_arm_counter_[iid][selected] = 0.0;
            }
            admission_arm_mask_[iid] = admission_requests_[iid] == 0 ? 0x0fU
                : static_cast<std::uint8_t>(1U << static_cast<unsigned>(selected));
            break;
        }
        case AdmissionPolicy::Choke: {
            const double threshold = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.50;
            const double gain = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 1.0;
            const double maximum = config_.admission.tertiary > 0.0
                ? config_.admission.tertiary : 0.95;
            const bool congested = admission_occupancy_ewma_[iid] >= 0.50
                && admission_requests_[iid] >= 2;
            for (std::size_t d = 0; d < 4; ++d) {
                const double share = admission_requests_[iid] == 0 ? 0.0
                    : static_cast<double>(admission_arm_requests_[iid][d])
                        / static_cast<double>(admission_requests_[iid]);
                admission_arm_probability_[iid][d] = !congested || share <= threshold ? 0.0
                    : std::min(maximum, gain * (share - threshold)
                        / std::max(1e-9, 1.0 - threshold));
            }
            break;
        }
        case AdmissionPolicy::QueueCsma: {
            const double exponent = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 1.0;
            double total_weight = 0.0;
            std::array<double, 4> weights{};
            for (std::size_t d = 0; d < 4; ++d) {
                if (admission_arm_requests_[iid][d] == 0) continue;
                weights[d] = std::pow(1.0 + static_cast<double>(admission_arm_requests_[iid][d]),
                                      exponent);
                total_weight += weights[d];
            }
            if (total_weight <= 0.0) break;
            std::uint64_t value = admission_seed_
                ^ ((stats_.timestep + 1U) * 0x9e3779b97f4a7c15ULL)
                ^ ((iid + 1U) * 0xbf58476d1ce4e5b9ULL);
            value ^= value >> 30U;
            value *= 0xbf58476d1ce4e5b9ULL;
            value ^= value >> 27U;
            value *= 0x94d049bb133111ebULL;
            value ^= value >> 31U;
            double draw = static_cast<double>(value >> 11U)
                * (1.0 / 9007199254740992.0) * total_weight;
            std::size_t selected = 0;
            for (std::size_t d = 0; d < 4; ++d) {
                if (draw < weights[d]) { selected = d; break; }
                draw -= weights[d];
            }
            admission_arm_mask_[iid] = static_cast<std::uint8_t>(
                1U << static_cast<unsigned>(selected));
            break;
        }
        case AdmissionPolicy::StochasticFairBlue: {
            const double increase = config_.admission.parameter > 0.0
                ? config_.admission.parameter : 0.02;
            const double decrease = config_.admission.secondary > 0.0
                ? config_.admission.secondary : 0.002;
            const std::uint32_t delay_target = static_cast<std::uint32_t>(
                config_.admission.tertiary > 0.0 ? config_.admission.tertiary : 3.0);
            for (std::size_t d = 0; d < 4; ++d) {
                if (admission_arm_max_wait_[iid][d] > delay_target)
                    admission_arm_probability_[iid][d] = std::min(
                        0.95, admission_arm_probability_[iid][d] + increase);
                else if (admission_arm_requests_[iid][d] == 0)
                    admission_arm_probability_[iid][d] = std::max(
                        0.0, admission_arm_probability_[iid][d] - decrease);
            }
            break;
        }
        case AdmissionPolicy::FqCodel: {
            const std::uint32_t target = static_cast<std::uint32_t>(
                config_.admission.parameter > 0.0 ? config_.admission.parameter : 3.0);
            const std::uint32_t interval = static_cast<std::uint32_t>(
                config_.admission.secondary > 0.0 ? config_.admission.secondary : 5.0);
            const double quantum = config_.admission.tertiary > 0.0
                ? config_.admission.tertiary : 1.0;
            std::uint32_t oldest = 0;
            for (const auto wait : admission_arm_max_wait_[iid]) oldest = std::max(oldest, wait);
            if (oldest > target && occupied > 0) ++admission_congestion_age_[iid];
            else admission_congestion_age_[iid] = 0;
            reserve = std::max(reserve, std::min(capacity - 1,
                static_cast<int>(admission_congestion_age_[iid] / std::max(1U, interval))));
            int selected = -1;
            for (std::size_t d = 0; d < 4; ++d)
                if (admission_arm_requests_[iid][d] != 0) admission_deficit_[iid][d] += quantum;
            for (std::size_t d = 0; d < 4; ++d) {
                if (admission_arm_requests_[iid][d] == 0) continue;
                if (selected < 0 || admission_deficit_[iid][d] > admission_deficit_[iid][selected])
                    selected = static_cast<int>(d);
            }
            admission_arm_mask_[iid] = selected < 0 ? 0x0fU
                : static_cast<std::uint8_t>(1U << static_cast<unsigned>(selected));
            break;
        }
        case AdmissionPolicy::LongestQueue:
        case AdmissionPolicy::OldestRequest:
        case AdmissionPolicy::RoundRobin:
        case AdmissionPolicy::DeficitRoundRobin: {
            int selected = -1;
            if (policy == AdmissionPolicy::RoundRobin) {
                for (std::size_t offset = 0; offset < 4; ++offset) {
                    const auto d = (admission_rr_cursor_[iid] + offset) % 4U;
                    if (admission_arm_requests_[iid][d] != 0) {
                        selected = static_cast<int>(d);
                        admission_rr_cursor_[iid] = (d + 1U) % 4U;
                        break;
                    }
                }
            } else if (policy == AdmissionPolicy::DeficitRoundRobin) {
                const double quantum = config_.admission.parameter > 0.0
                    ? config_.admission.parameter : 1.0;
                for (std::size_t d = 0; d < 4; ++d)
                    if (admission_arm_requests_[iid][d] != 0) admission_deficit_[iid][d] += quantum;
                for (std::size_t d = 0; d < 4; ++d) {
                    if (admission_arm_requests_[iid][d] == 0) continue;
                    if (selected < 0 || admission_deficit_[iid][d] > admission_deficit_[iid][selected])
                        selected = static_cast<int>(d);
                }
            } else {
                for (std::size_t d = 0; d < 4; ++d) {
                    if (admission_arm_requests_[iid][d] == 0) continue;
                    if (selected < 0) selected = static_cast<int>(d);
                    else if (policy == AdmissionPolicy::LongestQueue
                        && admission_arm_requests_[iid][d] > admission_arm_requests_[iid][selected])
                        selected = static_cast<int>(d);
                    else if (policy == AdmissionPolicy::OldestRequest
                        && std::pair(admission_arm_max_wait_[iid][d], admission_arm_requests_[iid][d])
                           > std::pair(admission_arm_max_wait_[iid][selected], admission_arm_requests_[iid][selected]))
                        selected = static_cast<int>(d);
                }
            }
            admission_arm_mask_[iid] = selected < 0 ? 0x0fU
                : static_cast<std::uint8_t>(1U << static_cast<unsigned>(selected));
            break;
        }
        }

        admission_reserve_[iid] = std::clamp(reserve, 0, capacity);
    }
}

bool Simulator::admission_policy_blocks(const CellId current, const IntersectionId entering,
                                        const std::size_t direction) const {
    const auto iid = static_cast<std::size_t>(entering);
    double probability = 0.0;
    switch (config_.admission.policy) {
    case AdmissionPolicy::Red: {
        const double minimum = config_.admission.parameter > 0.0
            ? config_.admission.parameter : 0.50;
        const double maximum = config_.admission.secondary > minimum
            ? config_.admission.secondary : 0.90;
        const double max_probability = config_.admission.tertiary > 0.0
            ? config_.admission.tertiary : 0.50;
        const double average = admission_occupancy_ewma_[iid];
        if (average <= minimum) return false;
        probability = average >= maximum ? 1.0
            : max_probability * (average - minimum) / (maximum - minimum);
        break;
    }
    case AdmissionPolicy::Blue:
    case AdmissionPolicy::Rem:
    case AdmissionPolicy::Pie:
        probability = admission_probability_[iid];
        break;
    case AdmissionPolicy::Choke:
    case AdmissionPolicy::StochasticFairBlue:
        if (direction >= 4) return false;
        probability = admission_arm_probability_[iid][direction];
        break;
    default:
        return false;
    }
    if (probability <= 0.0) return false;

    std::uint64_t value = admission_seed_ ^ (stats_.timestep * 0x9e3779b97f4a7c15ULL)
        ^ (static_cast<std::uint64_t>(current) * 0xbf58476d1ce4e5b9ULL)
        ^ (static_cast<std::uint64_t>(entering + 1) * 0x94d049bb133111ebULL);
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    const double sample = static_cast<double>(value >> 11U) * (1.0 / 9007199254740992.0);
    return sample < probability;
}

bool Simulator::block_intersection(const CellId current, const CellId next, const bool normal_only) const {
    const auto& current_memberships = topology_.memberships(current);
    const bool current_outside = current_memberships.empty();
    const bool current_center = topology_.is_center(current);
    if (!current_outside && !current_center) return false;

    const IntersectionId entering = entering_intersection(current, next);
    if (entering < 0) return false;
    const auto iid = static_cast<std::size_t>(entering);
    if (normal_only) {
        if (intersection_available_[iid] <= 0) return true;
        const AgentId requester = occupancy_[static_cast<std::size_t>(current)];
        const bool aimd = config_.admission.policy == AdmissionPolicy::Aimd;
        const bool acknowledged_grant = requester != kNoAgent
            && aimd_admission_.granted(requester, entering);
        if (aimd && !acknowledged_grant) return true;
        if (!aimd) {
            if (intersection_available_[iid] <= admission_reserve_[iid]) return true;
            const auto direction = static_cast<std::size_t>(topology_.intersections()[iid].direction_of(next));
            if (direction < 4 && (admission_arm_mask_[iid] & (1U << direction)) == 0) return true;
            if (config_.admission.policy == AdmissionPolicy::TokenBucket && admission_tokens_[iid] < 1.0)
                return true;
            if (admission_policy_blocks(current, entering, direction)) return true;
        }
    }

    const auto priority = [&](const IntersectionId iid) {
        return deadlock_active_[static_cast<std::size_t>(iid)] != 0
            ? deadlock_priority_[static_cast<std::size_t>(iid)] : deadlock_queue_.size();
    };
    const std::size_t current_priority = current_outside || current_memberships.empty()
        ? deadlock_queue_.size() : priority(current_memberships.front());
    return priority(entering) < current_priority;
}

void Simulator::update_available_on_move(const AgentId agent, const CellId current, const CellId next) {
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
        const auto index = static_cast<std::size_t>(iid);
        intersection_available_[index] = std::max(0, intersection_available_[index] - 1);
        if (config_.admission.policy == AdmissionPolicy::Aimd)
            aimd_admission_.acknowledge_entry(agent, iid);
        if (config_.admission.policy == AdmissionPolicy::TokenBucket)
            admission_tokens_[index] = std::max(0.0, admission_tokens_[index] - 1.0);
        if (config_.admission.policy == AdmissionPolicy::DeficitRoundRobin
            || config_.admission.policy == AdmissionPolicy::FqCodel) {
            const auto direction = static_cast<std::size_t>(topology_.intersections()[index].direction_of(next));
            if (direction < 4) admission_deficit_[index][direction] = std::max(
                0.0, admission_deficit_[index][direction] - 1.0);
        }
    }
}

void Simulator::count_zone_entries(const AgentId agent, const CellId current, const CellId next) {
    if (!metrics_ || current == next) return;
    const auto& from = topology_.memberships(current);
    for (const IntersectionId iid : topology_.memberships(next)) {
        if (std::find(from.begin(), from.end(), iid) != from.end()) continue;
        const CellId center = topology_.intersections()[static_cast<std::size_t>(iid)].center;
        metrics_->on_acquisition(stats_.timestep, agent, iid, cell_distance(map_, next, center));
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
    if (metrics_) {
        metrics_->on_route_mutation(
            stats_.timestep, agent.id, "local_schedule",
            scheduled.path.size() + bridge.size() - 2, rejoin,
            !bridge.empty() && bridge.back() == rejoin,
            !agent.route.empty() && agent.route.back() == agent.goal);
    }
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
    // Reference progress is task-level state, not an alias of the mutable
    // active-route cursor. Update it for every consumed route waypoint so
    // externally supplied routes with explicit waits remain well-defined.
    advance_reference_progress(agent);
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
    std::fill(execution_failed_.begin(), execution_failed_.end(), 0);

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
    update_admission_policy();
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
    if (config_.shuffle_order >= 0) {
        // W7: replace the least-loaded-first processing order with a fresh
        // seeded permutation every cycle; completion must not depend on it.
        std::shuffle(candidates_.begin(), candidates_.end(), shuffle_rng_);
    } else {
        std::sort(candidates_.begin(), candidates_.end(), [&](const IntersectionId lhs, const IntersectionId rhs) {
            const auto l = inside_counts_[static_cast<std::size_t>(lhs)];
            const auto r = inside_counts_[static_cast<std::size_t>(rhs)];
            return l != r ? l < r : lhs < rhs;
        });
    }

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
            for (const IntersectionId neighbor : intersection.neighbors) {
                if (neighbor < 0) continue;
                const CellId neighbor_center = topology_.intersections()[
                    static_cast<std::size_t>(neighbor)].center;
                metrics_->on_gate_signal(stats_.timestep, iid_value, neighbor,
                                         cell_distance(map_, intersection.center, neighbor_center));
            }
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
        if (metrics_) {
            const CellId center = topology_.intersections()[
                static_cast<std::size_t>(schedule.intersection)].center;
            for (const AgentId id : applied) {
                metrics_->on_broadcast(
                    stats_.timestep, schedule.intersection, id,
                    cell_distance(map_, center, agents_[static_cast<std::size_t>(id)].position));
            }
        }
        if (tracer_ && !applied.empty()) tracer_->add_schedule(schedule.intersection, std::move(applied));
    }

    if (config_.discharge_enabled) {
        auto discharge_events = discharge_.run({topology_, agents_, members_, stalled_, deadlock_active_, *planner_, rng_,
                                                &intersection_available_, &prev_inside_counts_});
        for (auto& event : discharge_events) {
            if (metrics_) {
                metrics_->on_discharge(stats_.timestep, event.intersection, event.agent_ids,
                                       event.agent_loop_cells, event.agent_loop_closed);
                for (std::size_t i = 0; i < event.agent_ids.size(); ++i) {
                    const AgentId id = event.agent_ids[i];
                    metrics_->note_discharged_agent(id);
                    const Agent& agent = agents_[static_cast<std::size_t>(id)];
                    const std::size_t loop_cells = i < event.agent_loop_cells.size()
                        ? event.agent_loop_cells[i] : 0;
                    const bool loop_closed = i < event.agent_loop_closed.size()
                        && event.agent_loop_closed[i] != 0;
                    metrics_->on_route_mutation(
                        stats_.timestep, id, "recirculation",
                        loop_cells > 0 ? loop_cells - 1 : 0, agent.position,
                        loop_closed, !agent.route.empty() && agent.route.back() == agent.goal);
                }
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
            if (next != agent.position && sample_command_failure(agent.id)) {
                ++agent.wait_steps;
                ++stats_.waits;
                agent.wait_reason = WaitReason::ExecutionFailure;
                continue;
            }
            occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
            update_available_on_move(agent.id, agent.position, next);
            count_zone_entries(agent.id, agent.position, next);
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
    if (config_.pibt_corridor) {
        compute_rescue_groups();
    } else {
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
    // Unaffected scheduled cohorts move atomically.  If any independently
    // sampled command is dropped, stop the whole cohort for this timestep so
    // no follower enters the failed agent's cell.
    if (config_.failure_probability > 0.0) {
        for (const IntersectionId iid_value : deadlock_queue_) {
            const auto iid = static_cast<std::size_t>(iid_value);
            if (blocked_[iid] || (iid < rescue_group_.size() && rescue_group_[iid] != 0)) continue;
            bool failed = false;
            for (const AgentId id : scheduled_members_[iid]) {
                const Agent& agent = agents_[static_cast<std::size_t>(id)];
                if (!agent.active || !agent.scheduled() || agent.intended_cell() == agent.position) continue;
                failed = sample_command_failure(id) || failed;
            }
            if (failed) blocked_[iid] = true;
        }
    }
    }
    for (Agent& agent : agents_) {
        if (!agent.active || !agent.scheduled()) continue;
        const auto iid = static_cast<std::size_t>(agent.schedule_group);
        if (iid < rescue_group_.size() && rescue_group_[iid] != 0) continue;
        if (iid < blocked_.size() && blocked_[iid]) {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = execution_failed_[static_cast<std::size_t>(agent.id)] != 0
                ? WaitReason::ExecutionFailure : WaitReason::ScheduleGroup;
            continue;
        }
        occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
        const CellId previous = agent.position;
        move_agent(agent);
        count_zone_entries(agent.id, previous, agent.position);
        occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
    }

    if (config_.rotation_enabled) rotate_blocked_cycles();

    for (Agent& agent : agents_) {
        if (!agent.active || agent.position != agent.goal) continue;
        const auto index = static_cast<std::size_t>(agent.id);
        const std::uint64_t service_steps = stats_.timestep - task_start_steps_[index];
        if (config_.goal_behavior == GoalBehavior::Disappear) {
            agent.active = false;
            ++stats_.completed;
            completion_steps_[index] = stats_.timestep;
            if (metrics_) metrics_->on_task_completion(stats_.timestep, agent.id, 1, service_steps);
            if (tracer_) tracer_->add_completion(agent.id);
        } else if (config_.goal_behavior == GoalBehavior::Stay) {
            if (!agent.reached) {
                agent.reached = true;
                ++stats_.completed;
                completion_steps_[index] = stats_.timestep;
                if (metrics_) metrics_->on_task_completion(stats_.timestep, agent.id, 1, service_steps);
                if (tracer_) tracer_->add_completion(agent.id);
            }
        } else {  // Lifelong: serve the task; the allocator hands out the next one
            if (!agent.awaiting_goal) {
                agent.awaiting_goal = true;
                ++agent.tasks_completed;
                ++stats_.completed;
                completion_steps_[index] = stats_.timestep;
                if (metrics_) metrics_->on_task_completion(
                    stats_.timestep, agent.id, agent.tasks_completed, service_steps);
                if (tracer_) tracer_->add_completion(agent.id);
            }
        }
    }
    if (goal_allocator_) assign_lifelong_goals();
    if (metrics_) {
        metrics_->observe_step(stats_.timestep, agents_);
        metrics_->flush_step(stats_.timestep);
    }
    if (tracer_) tracer_->flush_step(stats_.timestep, agents_);
    return true;
}

void Simulator::assign_lifelong_goals() {
    for (Agent& agent : agents_) {
        if (!agent.active || !agent.awaiting_goal) continue;

        // A task arrival is serviced in this same timestep, including when
        // the agent reached its goal in the middle of a committed
        // intersection schedule.  In that case the already-broadcast prefix
        // remains immutable and the fresh route starts at the prefix's final
        // cell.  This preserves cohort safety without introducing a parked
        // or goal-less timestep between lifelong tasks.
        const std::size_t anchor_index = std::min(
            agent.route_cursor + agent.scheduling_remaining,
            agent.route.empty() ? std::size_t{0} : agent.route.size() - 1);
        const CellId anchor = agent.route.empty() ? agent.position : agent.route[anchor_index];
        std::vector<CellId> suffix;
        const auto next_goal = goal_allocator_->reassign(
            agent.id, agent.goal, agent.position, occupancy_, [&](const CellId candidate) {
                suffix = plan_global(anchor, candidate);
                return !suffix.empty() && suffix.front() == anchor && suffix.back() == candidate;
            });
        if (!next_goal)
            throw std::runtime_error("lifelong goal pool has no reachable free task");
        agent.goal = *next_goal;
        if (agent.scheduled()) {
            agent.route.resize(anchor_index + 1);
            agent.route.insert(agent.route.end(), suffix.begin() + 1, suffix.end());
        } else {
            agent.route = std::move(suffix);
            agent.route_cursor = 0;
        }
        agent.reference_route.assign(
            agent.route.begin() + static_cast<std::ptrdiff_t>(agent.route_cursor),
            agent.route.end());
        agent.reference_cursor = 0;
        agent.wait_steps = 0;
        agent.wait_reason = WaitReason::None;
        agent.awaiting_goal = false;
        task_start_steps_[static_cast<std::size_t>(agent.id)] = stats_.timestep;
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
        advance_reference_progress(agent);
    }
    agent.wait_steps = 0;
    agent.wait_reason = WaitReason::None;
}

void Simulator::advance_reference_progress(Agent& agent) {
    if (agent.reference_route.empty() || agent.reference_cursor >= agent.reference_route.size()) return;
    if (agent.reference_route[agent.reference_cursor] == agent.position) {
        // Consume only consecutive explicit waits. A later revisit to the same
        // cell must not skip the intervening reference segment.
        while (agent.reference_cursor + 1 < agent.reference_route.size()
               && agent.reference_route[agent.reference_cursor + 1] == agent.position) {
            ++agent.reference_cursor;
        }
        return;
    }
    const auto begin = agent.reference_route.begin()
        + static_cast<std::ptrdiff_t>(agent.reference_cursor + 1);
    const auto found = std::find(begin, agent.reference_route.end(), agent.position);
    if (found != agent.reference_route.end()) {
        agent.reference_cursor = static_cast<std::size_t>(
            std::distance(agent.reference_route.begin(), found));
    }
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

// Rescue-group detection and atomic rotation (PIBT-corridor mode only).
// Replaces the plain blocked_ computation: a schedule cohort whose only
// obstruction is an already released member of the same schedule standing on
// a scheduled destination is rotated forward as one atomic chain instead of
// waiting forever (mixed scheduled/released standstill).
void Simulator::compute_rescue_groups() {
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
        bool command_failed = false;
        for (const AgentId id : members) {
            const Agent& agent = agents_[static_cast<std::size_t>(id)];
            if (next_for(agent) != agent.position)
                command_failed = sample_command_failure(id) || command_failed;
        }
        if (command_failed) {
            blocked_[iid] = true;
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
            const CellId previous = agent.position;
            if (agent.scheduled() || forced == kInvalidCell) {
                if (!agent.scheduled()) update_available_on_move(
                    agent.id, agent.position, agent.intended_cell());
                move_agent(agent);
            } else {
                update_available_on_move(agent.id, agent.position, forced);
                move_agent_to(agent, forced);
            }
            count_zone_entries(agent.id, previous, agent.position);
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        }
    }
}

// PIBT-corridor movement phase (opt-in): agents outside managed intersections
// move via priority-inheritance backtracking; agents inside keep the original
// single-route semantics except for constrained boundary-exit roots.
void Simulator::run_pibt_movement() {
    // Route-Planner recovery: reconnect a displaced agent to its first
    // unfinished task-level reference waypoint. The bridge may change, but
    // the reference suffix and its monotone completion rank never do.
    if (config_.pibt_replan > 0) {
        for (Agent& agent : agents_) {
            if (!agent.active || agent.scheduled()) continue;
            if (agent.wait_steps < config_.pibt_replan
                || agent.wait_steps % config_.pibt_replan != 0) continue;
            const bool off_route = agent.route_cursor >= agent.route.size()
                || agent.route[agent.route_cursor] != agent.position;
            if (!off_route) continue;
            if (active_discharge_target(agent) != kInvalidCell) continue;
            auto repair = repair_to_reference_suffix(
                *planner_, agent.position, agent.reference_route, agent.reference_cursor);
            if (!repair || repair->route.empty() || repair->route.back() != agent.goal) continue;
            if (metrics_) {
                metrics_->on_route_mutation(
                    stats_.timestep, agent.id, "route_repair",
                    repair->bridge_edges, repair->rejoin, true,
                    repair->route.back() == agent.goal);
            }
            agent.route = std::move(repair->route);
            agent.route_cursor = 0;
        }
    }

    // Snapshot pre-movement origins: the rescue pass later this step must
    // verify a released dependency has not already moved via PIBT.
    for (const Agent& agent : agents_) if (agent.active)
        movement_origin_[static_cast<std::size_t>(agent.id)] = agent.position;

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

    // Sink-yield (opt-in straggler fix): a normal agent about to enter its own
    // goal cell while a differently-goaled agent stands on it is demoted below
    // every normal agent, so the occupant's dependency chain becomes the PIBT
    // root and can push its way free instead of being frozen by the swap rule.
    // Classes shift up one level so their relative order is preserved.
    if (config_.pibt_sink_yield) {
        for (const Agent& agent : agents_) {
            const auto index = static_cast<std::size_t>(agent.id);
            if (pibt_eligible_[index] == 0) continue;
            const std::uint8_t klass = pibt_priority_class_[index];
            bool yields = false;
            if (klass == 0 && agent.active && agent.intended_cell() == agent.goal) {
                const AgentId occupant = occupancy_[static_cast<std::size_t>(agent.goal)];
                if (occupant != kNoAgent && occupant != agent.id) {
                    const Agent& other = agents_[static_cast<std::size_t>(occupant)];
                    yields = other.active && other.goal != agent.goal;
                }
            }
            pibt_priority_class_[index] = yields ? 0 : static_cast<std::uint8_t>(klass + 1);
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
            if (next != agent.position && sample_command_failure(agent.id)) {
                ++agent.wait_steps;
                ++stats_.waits;
                agent.wait_reason = WaitReason::ExecutionFailure;
                continue;
            }
            occupancy_[static_cast<std::size_t>(agent.position)] = kNoAgent;
            update_available_on_move(agent.id, agent.position, next);
            count_zone_entries(agent.id, agent.position, next);
            move_agent(agent);
            occupancy_[static_cast<std::size_t>(agent.position)] = agent.id;
        } else {
            ++agent.wait_steps;
            ++stats_.waits;
            agent.wait_reason = WaitReason::Dependency;
        }
    }

    pibt_->resolve(agents_, occupancy_, pibt_eligible_, pibt_priority_class_,
        [&](const AgentId id, const CellId candidate, const bool relaxed) {
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
                && !rejoins_current_waypoint) {
                const bool retreat_allowed = config_.pibt_arm_retreat
                    || (config_.pibt_arm_retreat_last && relaxed);
                if (!retreat_allowed) return false;
                // Arm-retreat (opt-in straggler fix): permit the detour into
                // an arm cell (never a center) of intersections without an
                // active schedule; the admission gate below still applies.
                // Restores original-PIBT pushability at aisle mouths whose
                // only free sidestep is an arm cell.
                if (topology_.is_center(candidate)) return false;
                for (const IntersectionId iid : topology_.memberships(candidate))
                    if (deadlock_active_[static_cast<std::size_t>(iid)] != 0) return false;
            }
            return !block_intersection(agent.position, candidate, true);
        }, pibt_next_);

    if (config_.failure_probability > 0.0) {
        std::vector<std::uint8_t> cancelled(agents_.size(), 0);
        for (const Agent& agent : agents_) {
            const auto index = static_cast<std::size_t>(agent.id);
            if (pibt_eligible_[index] == 0 || pibt_next_[index] == kInvalidCell
                || pibt_next_[index] == agent.position) continue;
            if (sample_command_failure(agent.id)) cancelled[index] = 1;
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (const Agent& agent : agents_) {
                const auto index = static_cast<std::size_t>(agent.id);
                if (pibt_eligible_[index] == 0 || cancelled[index] != 0
                    || pibt_next_[index] == kInvalidCell || pibt_next_[index] == agent.position) continue;
                const AgentId occupant = occupancy_[static_cast<std::size_t>(pibt_next_[index])];
                if (occupant != kNoAgent && cancelled[static_cast<std::size_t>(occupant)] != 0) {
                    cancelled[index] = 1;
                    changed = true;
                }
            }
        }
        for (const Agent& agent : agents_) {
            const auto index = static_cast<std::size_t>(agent.id);
            if (cancelled[index] != 0) pibt_next_[index] = agent.position;
        }
    }

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
            agent.wait_reason = execution_failed_[index] != 0
                ? WaitReason::ExecutionFailure : WaitReason::Dependency;
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
        update_available_on_move(agent.id, agent.position, next);
        count_zone_entries(agent.id, agent.position, next);
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
            update_available_on_move(agent.id, previous, agent.intended_cell());
            count_zone_entries(agent.id, previous, agent.intended_cell());
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
        if (config_.goal_behavior == GoalBehavior::Lifelong && agent.awaiting_goal) {
            problem << "lifelong agent " << agent.id << " has no next goal after task completion";
            return problem.str();
        }
        if (agent.route_cursor >= agent.route.size() || agent.route[agent.route_cursor] != agent.position) {
            problem << "agent " << agent.id << " desynchronized from its route cursor";
            return problem.str();
        }
        if (agent.reference_route.empty()
            || agent.reference_cursor >= agent.reference_route.size()
            || agent.reference_route.back() != agent.goal) {
            problem << "agent " << agent.id << " has an invalid task-level reference suffix";
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
    if (map_.terminal(goal)) {
        std::vector<CellId> parent(static_cast<std::size_t>(map_.cell_count()), kInvalidCell);
        std::deque<CellId> queue{goal};
        parent[static_cast<std::size_t>(goal)] = goal;
        CellId gateway = kInvalidCell;
        while (!queue.empty() && gateway == kInvalidCell) {
            const CellId current = queue.front();
            queue.pop_front();
            for (const CellId neighbor : map_.neighbors(current)) {
                if (!map_.terminal(neighbor) && !map_.goal(neighbor)) continue;
                if (parent[static_cast<std::size_t>(neighbor)] != kInvalidCell) continue;
                parent[static_cast<std::size_t>(neighbor)] = current;
                if (map_.goal(neighbor)) {
                    gateway = neighbor;
                    break;
                }
                queue.push_back(neighbor);
            }
        }
        if (gateway != kInvalidCell) {
            auto route = plan_global(start, gateway);
            if (route.empty()) return {};
            for (CellId current = gateway; current != goal;) {
                current = parent[static_cast<std::size_t>(current)];
                if (current == kInvalidCell) return {};
                route.push_back(current);
            }
            return route;
        }
    }
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
    if (map_.goal(goal)) {
        for (const CellId neighbor : map_.neighbors(goal)) if (map_.terminal(neighbor)) {
            const Coord terminal = map_.coord(neighbor);
            vertical_goal = vertical_goal || terminal.y != destination.y;
            horizontal_goal = horizontal_goal || terminal.x != destination.x;
        }
    }
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
