#include "lima/io/debug_trace.hpp"

#include "lima/core/agent.hpp"
#include "lima/core/grid_map.hpp"
#include "lima/simulation/simulator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace lima {
namespace {

std::ofstream open_output(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write debug log: " + path.string());
    return output;
}

void json_string(std::ostream& output, const std::string_view value) {
    output << '"';
    for (const char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << ch; break;
        }
    }
    output << '"';
}

void json_cell(std::ostream& output, const GridMap& map, const CellId cell) {
    if (cell == kInvalidCell || !map.traversable(cell)) {
        output << "null";
        return;
    }
    const Coord coordinate = map.coord(cell);
    output << '[' << coordinate.x << ',' << coordinate.y << ']';
}

void json_cells(std::ostream& output, const GridMap& map, const std::span<const CellId> cells) {
    output << '[';
    for (std::size_t index = 0; index < cells.size(); ++index) {
        if (index != 0) output << ',';
        json_cell(output, map, cells[index]);
    }
    output << ']';
}

template <typename Range>
void json_numbers(std::ostream& output, const Range& values) {
    output << '[';
    bool first = true;
    for (const auto value : values) {
        if (!first) output << ',';
        first = false;
        output << value;
    }
    output << ']';
}

const char* planner_name(const PlannerKind planner) {
    return planner == PlannerKind::AStar ? "astar" : "bfs";
}

const char* direction_name(const Direction direction) {
    switch (direction) {
        case Direction::North: return "north";
        case Direction::East: return "east";
        case Direction::South: return "south";
        case Direction::West: return "west";
        case Direction::Center: return "center";
        case Direction::None: return "none";
    }
    return "unknown";
}

const char* wait_reason_name(const WaitReason reason) {
    switch (reason) {
        case WaitReason::None: return "none";
        case WaitReason::ScheduledHold: return "scheduled_hold";
        case WaitReason::IntersectionReserved: return "intersection_reserved";
        case WaitReason::IntersectionCapacity: return "intersection_capacity";
        case WaitReason::VertexConflict: return "vertex_conflict";
        case WaitReason::EdgeSwap: return "edge_swap";
        case WaitReason::Dependency: return "dependency";
        case WaitReason::ScheduleGroup: return "schedule_group";
    }
    return "unknown";
}

const char* schedule_decision_name(const ScheduleDecision decision) {
    switch (decision) {
        case ScheduleDecision::NotChecked: return "not_checked";
        case ScheduleDecision::Active: return "active";
        case ScheduleDecision::NoDeadlock: return "no_deadlock";
        case ScheduleDecision::NeighborActive: return "neighbor_active";
        case ScheduleDecision::Candidate: return "candidate";
        case ScheduleDecision::CapacityExceeded: return "capacity_exceeded";
        case ScheduleDecision::NeighborQuotaBlocked: return "neighbor_quota_blocked";
        case ScheduleDecision::PlannerFailed: return "planner_failed";
        case ScheduleDecision::Activated: return "activated";
    }
    return "unknown";
}

std::size_t route_hash(const std::span<const CellId> route) {
    std::size_t hash = sizeof(std::size_t) == 8
        ? static_cast<std::size_t>(1469598103934665603ULL)
        : static_cast<std::size_t>(2166136261U);
    constexpr std::size_t prime64 = static_cast<std::size_t>(1099511628211ULL);
    constexpr std::size_t prime32 = static_cast<std::size_t>(16777619U);
    const std::size_t prime = sizeof(std::size_t) == 8 ? prime64 : prime32;
    for (const CellId cell : route) {
        hash ^= static_cast<std::size_t>(static_cast<std::uint32_t>(cell));
        hash *= prime;
    }
    return hash;
}

bool adjacent(const GridMap& map, const CellId lhs, const CellId rhs) {
    if (lhs == rhs) return true;
    const auto neighbors = map.neighbors(lhs);
    return std::find(neighbors.begin(), neighbors.end(), rhs) != neighbors.end();
}

Direction intended_exit(const Intersection& intersection, const Agent& agent) {
    Direction exit = intersection.direction_of(agent.position);
    for (std::size_t index = agent.route_cursor; index + 1 < agent.route.size(); ++index) {
        if (agent.route[index] != intersection.center) continue;
        const Direction candidate = intersection.direction_of(agent.route[index + 1]);
        if (candidate != Direction::None && candidate != Direction::Center) exit = candidate;
        break;
    }
    return exit;
}

}  // namespace

DebugTrace::DebugTrace(const std::filesystem::path& directory, const Simulator& simulator,
                       std::string map_file, std::string scenario_file, std::string command_line,
                       const std::uint64_t seed,
                       const PlannerKind planner, const std::uint64_t max_steps)
    : directory_(directory) {
    std::filesystem::create_directories(directory_);
    steps_ = open_output(directory_ / "steps.jsonl");
    agents_ = open_output(directory_ / "agents.jsonl");
    intersections_ = open_output(directory_ / "intersections.jsonl");
    schedules_ = open_output(directory_ / "schedules.jsonl");
    routes_ = open_output(directory_ / "routes.jsonl");
    events_ = open_output(directory_ / "events.jsonl");
    anomalies_ = open_output(directory_ / "anomalies.jsonl");

    const std::size_t agent_count = simulator.agents_.size();
    const std::size_t intersection_count = simulator.topology_.intersections().size();
    previous_positions_.assign(agent_count, kInvalidCell);
    previous_goals_.assign(agent_count, kInvalidCell);
    previous_route_hashes_.assign(agent_count, std::numeric_limits<std::size_t>::max());
    previous_scheduling_.assign(agent_count, std::numeric_limits<std::size_t>::max());
    previous_schedule_groups_.assign(agent_count, std::numeric_limits<std::int32_t>::min());
    previous_wait_steps_.assign(agent_count, std::numeric_limits<std::uint32_t>::max());
    previous_completed_.assign(agent_count, 2);
    previous_intersection_active_.assign(intersection_count, 2);
    previous_intersection_waiting_.assign(intersection_count, 2);

    write_metadata(simulator, map_file, scenario_file, command_line, seed, planner, max_steps);
    write_schema();
    {
        std::ofstream summary = open_output(directory_ / "summary.json");
        summary << "{\"status\":\"running\"}\n";
    }
    append(simulator, "initial");
}

void DebugTrace::write_schema() const {
    std::ofstream output = open_output(directory_ / "README.txt");
    output
        << "LIMA debug trace (JSON Lines; one JSON object per line)\n\n"
        << "metadata.json       Run options, map dimensions, and static intersection topology.\n"
        << "steps.jsonl         One summary record per simulation frame.\n"
        << "agents.jsonl        Every AMR state and movement decision for every frame.\n"
        << "intersections.jsonl Every intersection state for every frame.\n"
        << "schedules.jsonl     Full paths produced by newly activated schedules.\n"
        << "routes.jsonl        Full AMR route whenever the route itself changes.\n"
        << "events.jsonl        State transitions: schedule activation/release, waits, completion.\n"
        << "anomalies.jsonl     Vertex/edge conflicts and violated simulator invariants.\n"
        << "summary.json        Final status and aggregate counters.\n\n"
        << "All coordinates use [x,y]. Fields ending in _before_move are the scheduler's\n"
        << "snapshot at the beginning of that timestep; members_after_move and AMR positions\n"
        << "describe the committed end-of-timestep state.\n";
}

void DebugTrace::write_metadata(const Simulator& simulator, const std::string_view map_file,
                                const std::string_view scenario_file,
                                const std::string_view command_line,
                                const std::uint64_t seed, const PlannerKind planner,
                                const std::uint64_t max_steps) {
    std::ofstream output = open_output(directory_ / "metadata.json");
    output << "{\"format_version\":1,\"map_file\":";
    json_string(output, map_file);
    output << ",\"scenario_file\":";
    if (scenario_file.empty()) output << "null";
    else json_string(output, scenario_file);
    output << ",\"command_line\":";
    json_string(output, command_line);
    output << ",\"seed\":" << seed
           << ",\"planner\":\"" << planner_name(planner) << "\""
           << ",\"workload\":\"" << (simulator.lifelong() ? "lifelong" : "oneshot") << "\""
           << ",\"max_steps\":" << max_steps
           << ",\"agent_count\":" << simulator.agents_.size()
           << ",\"map\":{\"width\":" << simulator.map_.width()
           << ",\"height\":" << simulator.map_.height() << "}"
           << ",\"intersection_count\":" << simulator.topology_.intersections().size()
           << ",\"topology\":[";
    bool first_intersection = true;
    for (const Intersection& intersection : simulator.topology_.intersections()) {
        if (!first_intersection) output << ',';
        first_intersection = false;
        output << "{\"id\":" << intersection.id << ",\"center\":";
        json_cell(output, simulator.map_, intersection.center);
        output << ",\"neighbors\":";
        json_numbers(output, intersection.neighbors);
        output << ",\"arms\":[";
        for (std::size_t direction = 0; direction < 4; ++direction) {
            if (direction != 0) output << ',';
            json_cells(output, simulator.map_, intersection.arms[direction]);
        }
        output << "]}";
    }
    output << "]}\n";
}

void DebugTrace::append(const Simulator& simulator, const std::string_view phase) {
    const GridMap& map = simulator.map_;
    const auto& agents = simulator.agents_;
    const auto timestep = simulator.stats_.timestep;
    const std::vector<CellId> frame_previous = previous_positions_;

    std::vector<std::vector<AgentId>> members_after(simulator.topology_.intersections().size());
    std::vector<AgentId> current_occupancy(static_cast<std::size_t>(map.cell_count()), kNoAgent);
    std::size_t active_agents = 0;
    for (const Agent& agent : agents) {
        if (!agent.active) continue;
        ++active_agents;
        current_occupancy[static_cast<std::size_t>(agent.position)] = agent.id;
        for (const IntersectionId iid : simulator.topology_.memberships(agent.position))
            members_after[static_cast<std::size_t>(iid)].push_back(agent.id);
    }

    steps_ << "{\"timestep\":" << timestep << ",\"phase\":";
    json_string(steps_, phase);
    steps_ << ",\"completed\":" << simulator.stats_.completed
           << ",\"tasks_completed\":" << simulator.stats_.completed_tasks
           << ",\"active_agents\":" << active_agents
           << ",\"committed_moves\":" << simulator.stats_.committed_moves
           << ",\"waits\":" << simulator.stats_.waits
           << ",\"detected_deadlocks\":" << simulator.stats_.detected_deadlocks
           << ",\"deadlock_queue\":";
    json_numbers(steps_, simulator.deadlock_queue_);
    steps_ << ",\"candidates\":";
    json_numbers(steps_, simulator.candidates_);
    steps_ << ",\"new_schedule_intersections\":[";
    for (std::size_t index = 0; index < simulator.pending_.size(); ++index) {
        if (index != 0) steps_ << ',';
        steps_ << simulator.pending_[index].intersection;
    }
    steps_ << "]}\n";

    for (const auto& schedule : simulator.pending_) {
        for (const ScheduledPath& path : schedule.paths) {
            schedules_ << "{\"timestep\":" << timestep
                       << ",\"intersection\":" << schedule.intersection
                       << ",\"agent\":" << path.agent << ",\"target_exit\":";
            json_cell(schedules_, map, path.target_exit);
            schedules_ << ",\"path\":";
            json_cells(schedules_, map, path.path);
            schedules_ << "}\n";
        }
    }

    for (const Agent& agent : agents) {
        const std::size_t index = static_cast<std::size_t>(agent.id);
        const CellId previous = frame_previous[index];
        const CellId forced = index < simulator.pibt_forced_next_.size()
            ? simulator.pibt_forced_next_[index] : kInvalidCell;
        const CellId resolved = index < simulator.pibt_next_.size()
            ? simulator.pibt_next_[index] : kInvalidCell;
        const CellId decision_origin = first_frame_ ? agent.position : index < simulator.movement_origin_.size()
            ? simulator.movement_origin_[index] : agent.position;
        const CellId decision_intended = first_frame_ ? agent.intended_cell() : index < simulator.movement_intended_.size()
            ? simulator.movement_intended_[index] : agent.intended_cell();
        agents_ << "{\"timestep\":" << timestep << ",\"id\":" << agent.id
                << ",\"position\":";
        json_cell(agents_, map, agent.position);
        agents_ << ",\"previous_position\":";
        json_cell(agents_, map, previous);
        agents_ << ",\"decision_origin\":";
        json_cell(agents_, map, decision_origin);
        agents_ << ",\"decision_intended\":";
        json_cell(agents_, map, decision_intended);
        agents_ << ",\"moved\":" << (!first_frame_ && previous != agent.position ? "true" : "false")
                << ",\"goal\":";
        json_cell(agents_, map, agent.goal);
        agents_ << ",\"intended_cell\":";
        json_cell(agents_, map, agent.intended_cell());
        agents_ << ",\"route_cursor\":" << agent.route_cursor
                << ",\"route_size\":" << agent.route.size()
                << ",\"scheduling_before_move\":"
                << (first_frame_ ? agent.scheduling_remaining
                    : index < simulator.movement_scheduling_.size() ? simulator.movement_scheduling_[index] : 0)
                << ",\"wait_steps_before_move\":"
                << (first_frame_ ? agent.wait_steps
                    : index < simulator.movement_wait_steps_.size() ? simulator.movement_wait_steps_[index] : 0)
                << ",\"scheduling_remaining\":" << agent.scheduling_remaining
                << ",\"schedule_group\":" << agent.schedule_group
                << ",\"scheduled\":" << (agent.scheduled() ? "true" : "false")
                << ",\"active\":" << (agent.active ? "true" : "false")
                << ",\"completed\":" << (agent.completed ? "true" : "false")
                << ",\"awaiting_goal\":" << (agent.awaiting_goal ? "true" : "false")
                << ",\"tasks_completed\":" << agent.tasks_completed
                << ",\"wait_steps\":" << agent.wait_steps
                << ",\"wait_reason\":\"" << wait_reason_name(agent.wait_reason) << "\""
                << ",\"moves\":" << agent.moves
                << ",\"memberships\":";
        json_numbers(agents_, simulator.topology_.memberships(agent.position));
        agents_ << ",\"intended_occupant\":";
        if (map.traversable(agent.intended_cell())) {
            const AgentId occupant = current_occupancy[static_cast<std::size_t>(agent.intended_cell())];
            if (occupant == kNoAgent) agents_ << "null";
            else agents_ << occupant;
        } else agents_ << "null";
        agents_ << ",\"intersection_intents\":[";
        bool first_intent = true;
        for (const IntersectionId iid_value : simulator.topology_.memberships(agent.position)) {
            if (!first_intent) agents_ << ',';
            first_intent = false;
            const Intersection& intersection = simulator.topology_.intersections()[static_cast<std::size_t>(iid_value)];
            agents_ << "{\"intersection\":" << iid_value << ",\"current\":\""
                    << direction_name(intersection.direction_of(agent.position)) << "\",\"exit\":\""
                    << direction_name(intended_exit(intersection, agent)) << "\"}";
        }
        agents_ << ']';
        agents_ << ",\"pibt\":{\"eligible\":"
                << (index < simulator.pibt_eligible_.size() && simulator.pibt_eligible_[index] != 0 ? "true" : "false")
                << ",\"priority_class\":"
                << (index < simulator.pibt_priority_class_.size()
                    ? static_cast<int>(simulator.pibt_priority_class_[index]) : 0)
                << ",\"forced_next\":";
        json_cell(agents_, map, forced);
        agents_ << ",\"resolved_next\":";
        json_cell(agents_, map, resolved);
        agents_ << "},\"rescue_group\":"
                << (index < simulator.rescue_member_.size() ? simulator.rescue_member_[index] : kNoGroup)
                << "}\n";

        const std::size_t hash = route_hash(agent.route);
        if (hash != previous_route_hashes_[index]) {
            routes_ << "{\"timestep\":" << timestep << ",\"agent\":" << agent.id
                    << ",\"reason\":\"" << (first_frame_ ? "initial" : "route_changed") << "\""
                    << ",\"route_cursor\":" << agent.route_cursor << ",\"route\":";
            json_cells(routes_, map, agent.route);
            routes_ << "}\n";
            previous_route_hashes_[index] = hash;
        }

        if (!first_frame_) {
            if (previous_goals_[index] != agent.goal) {
                events_ << "{\"timestep\":" << timestep
                        << ",\"type\":\"goal_assigned\",\"agent\":" << agent.id
                        << ",\"from\":";
                json_cell(events_, map, previous_goals_[index]);
                events_ << ",\"to\":";
                json_cell(events_, map, agent.goal);
                events_ << "}\n";
            }
            if (previous_scheduling_[index] == 0 && agent.scheduling_remaining > 0)
                events_ << "{\"timestep\":" << timestep << ",\"type\":\"agent_schedule_started\",\"agent\":"
                        << agent.id << ",\"group\":" << agent.schedule_group << "}\n";
            if (previous_scheduling_[index] > 0 && agent.scheduling_remaining == 0)
                events_ << "{\"timestep\":" << timestep << ",\"type\":\"agent_schedule_finished\",\"agent\":"
                        << agent.id << ",\"group\":" << agent.schedule_group << "}\n";
            if (previous_schedule_groups_[index] != agent.schedule_group)
                events_ << "{\"timestep\":" << timestep << ",\"type\":\"agent_group_changed\",\"agent\":"
                        << agent.id << ",\"from\":" << previous_schedule_groups_[index]
                        << ",\"to\":" << agent.schedule_group << "}\n";
            if (previous_completed_[index] != static_cast<std::uint8_t>(agent.completed))
                events_ << "{\"timestep\":" << timestep << ",\"type\":\"agent_completion_changed\",\"agent\":"
                        << agent.id << ",\"completed\":" << (agent.completed ? "true" : "false") << "}\n";
            constexpr std::uint32_t thresholds[]{1, 10, 50, 100, 500};
            for (const std::uint32_t threshold : thresholds) {
                if (previous_wait_steps_[index] < threshold && agent.wait_steps >= threshold) {
                    events_ << "{\"timestep\":" << timestep << ",\"type\":\"agent_wait_threshold\",\"agent\":"
                            << agent.id << ",\"threshold\":" << threshold << ",\"reason\":\""
                            << wait_reason_name(agent.wait_reason) << "\"}\n";
                }
            }
        }
        previous_positions_[index] = agent.position;
        previous_goals_[index] = agent.goal;
        previous_scheduling_[index] = agent.scheduling_remaining;
        previous_schedule_groups_[index] = agent.schedule_group;
        previous_wait_steps_[index] = agent.wait_steps;
        previous_completed_[index] = static_cast<std::uint8_t>(agent.completed);
    }

    for (const Intersection& intersection : simulator.topology_.intersections()) {
        const std::size_t iid = static_cast<std::size_t>(intersection.id);
        std::vector<AgentId> scheduled_members(
            simulator.scheduled_members_[iid].begin(), simulator.scheduled_members_[iid].end());
        std::sort(scheduled_members.begin(), scheduled_members.end());
        intersections_ << "{\"timestep\":" << timestep << ",\"id\":" << intersection.id
                       << ",\"center\":";
        json_cell(intersections_, map, intersection.center);
        intersections_ << ",\"active\":" << (simulator.deadlock_active_[iid] != 0 ? "true" : "false")
                       << ",\"waiting\":" << (simulator.deadlock_waiting_[iid] ? "true" : "false")
                       << ",\"schedule_decision\":\""
                       << schedule_decision_name(simulator.schedule_decision_[iid]) << "\""
                       << ",\"release_grace\":" << static_cast<int>(simulator.deadlock_release_grace_[iid])
                       << ",\"queue_priority\":";
        if (simulator.deadlock_active_[iid] != 0) intersections_ << simulator.deadlock_priority_[iid];
        else intersections_ << "null";
        intersections_ << ",\"available\":" << simulator.intersection_available_[iid]
                       << ",\"capacity\":" << simulator.intersection_capacity_[iid]
                       << ",\"inside_count_before_move\":" << simulator.inside_counts_[iid]
                       << ",\"check_requested\":" << (simulator.check_[iid] ? "true" : "false")
                       << ",\"stalled_before_move\":" << (simulator.stalled_[iid] ? "true" : "false")
                       << ",\"movement_blocked\":" << (simulator.blocked_[iid] ? "true" : "false")
                       << ",\"rescue_candidate\":" << (simulator.rescue_candidate_[iid] != 0 ? "true" : "false")
                       << ",\"rescue_committed\":" << (simulator.rescue_group_[iid] != 0 ? "true" : "false")
                       << ",\"quota_snapshot_valid\":"
                       << (simulator.debug_quota_valid_[iid] != 0 ? "true" : "false")
                       << ",\"initial_counts\":";
        if (simulator.debug_quota_valid_[iid] != 0)
            json_numbers(intersections_, simulator.debug_initial_counts_[iid]);
        else intersections_ << "null";
        intersections_ << ",\"quotas\":";
        if (simulator.debug_quota_valid_[iid] != 0)
            json_numbers(intersections_, simulator.debug_quotas_[iid]);
        else intersections_ << "null";
        intersections_ << ",\"predicted_final_counts\":";
        if (simulator.debug_final_valid_[iid] != 0)
            json_numbers(intersections_, simulator.debug_final_counts_[iid]);
        else intersections_ << "null";
        intersections_
                       << ",\"members_before_move\":";
        json_numbers(intersections_, simulator.members_[iid]);
        intersections_ << ",\"members_after_move\":";
        json_numbers(intersections_, members_after[iid]);
        intersections_ << ",\"scheduled_members\":";
        json_numbers(intersections_, scheduled_members);
        intersections_ << ",\"intent_snapshot_valid\":"
                       << (simulator.intent_valid_[iid] != 0 ? "true" : "false")
                       << ",\"intents\":[";
        if (simulator.intent_valid_[iid] != 0) {
            for (std::size_t index = 0; index < simulator.intents_[iid].size(); ++index) {
                if (index != 0) intersections_ << ',';
                const IntersectionIntent& intent = simulator.intents_[iid][index];
                intersections_ << "{\"agent\":" << intent.agent << ",\"current\":\""
                               << direction_name(intent.current) << "\",\"exit\":\""
                               << direction_name(intent.exit) << "\",\"position\":";
                json_cell(intersections_, map, intent.position);
                intersections_ << ",\"next\":";
                json_cell(intersections_, map, intent.next);
                intersections_ << '}';
            }
        }
        intersections_ << "]}\n";

        const std::uint8_t active = simulator.deadlock_active_[iid] != 0;
        const std::uint8_t waiting = simulator.deadlock_waiting_[iid] ? 1 : 0;
        if (!first_frame_ && previous_intersection_active_[iid] != active)
            events_ << "{\"timestep\":" << timestep << ",\"type\":\"intersection_"
                    << (active != 0 ? "activated" : "released") << "\",\"intersection\":"
                    << intersection.id << "}\n";
        if (!first_frame_ && previous_intersection_waiting_[iid] != waiting)
            events_ << "{\"timestep\":" << timestep << ",\"type\":\"intersection_waiting_changed\",\"intersection\":"
                    << intersection.id << ",\"waiting\":" << (waiting != 0 ? "true" : "false") << "}\n";
        previous_intersection_active_[iid] = active;
        previous_intersection_waiting_[iid] = waiting;
    }

    std::vector<AgentId> occupancy(static_cast<std::size_t>(map.cell_count()), kNoAgent);
    for (const Agent& agent : agents) {
        if (!agent.active) continue;
        if (!map.traversable(agent.position)) {
            anomalies_ << "{\"timestep\":" << timestep << ",\"type\":\"invalid_position\",\"agent\":"
                       << agent.id << ",\"cell_id\":" << agent.position << "}\n";
            ++anomaly_count_;
            continue;
        }
        AgentId& occupant = occupancy[static_cast<std::size_t>(agent.position)];
        if (occupant != kNoAgent) {
            anomalies_ << "{\"timestep\":" << timestep << ",\"type\":\"vertex_conflict\",\"agents\":["
                       << occupant << ',' << agent.id << "],\"cell\":";
            json_cell(anomalies_, map, agent.position);
            anomalies_ << "}\n";
            ++anomaly_count_;
        } else occupant = agent.id;
        if (!first_frame_ && !adjacent(map, frame_previous[static_cast<std::size_t>(agent.id)], agent.position)) {
            anomalies_ << "{\"timestep\":" << timestep << ",\"type\":\"non_adjacent_move\",\"agent\":"
                       << agent.id << ",\"from\":";
            json_cell(anomalies_, map, frame_previous[static_cast<std::size_t>(agent.id)]);
            anomalies_ << ",\"to\":";
            json_cell(anomalies_, map, agent.position);
            anomalies_ << "}\n";
            ++anomaly_count_;
        }
        if (agent.scheduled()) {
            const auto group = static_cast<std::size_t>(agent.schedule_group);
            if (agent.schedule_group < 0 || group >= simulator.deadlock_active_.size()
                || simulator.deadlock_active_[group] == 0) {
                anomalies_ << "{\"timestep\":" << timestep
                           << ",\"type\":\"scheduled_agent_without_active_group\",\"agent\":"
                           << agent.id << ",\"group\":" << agent.schedule_group << "}\n";
                ++anomaly_count_;
            }
        }
    }
    if (!first_frame_) {
        std::vector<AgentId> previous_occupancy(static_cast<std::size_t>(map.cell_count()), kNoAgent);
        for (const Agent& agent : agents) if (agent.active) {
            const std::size_t index = static_cast<std::size_t>(agent.id);
            if (map.traversable(frame_previous[index]))
                previous_occupancy[static_cast<std::size_t>(frame_previous[index])] = agent.id;
        }
        for (const Agent& agent : agents) {
            const std::size_t lhs = static_cast<std::size_t>(agent.id);
            if (!agent.active || frame_previous[lhs] == agent.position) continue;
            const AgentId other = previous_occupancy[static_cast<std::size_t>(agent.position)];
            if (other <= agent.id) continue;
            const std::size_t rhs = static_cast<std::size_t>(other);
            if (rhs < agents.size() && agents[rhs].active
                && agents[rhs].position == frame_previous[lhs]) {
                anomalies_ << "{\"timestep\":" << timestep << ",\"type\":\"edge_conflict\",\"agents\":["
                           << lhs << ',' << rhs << "],\"edge\":[";
                json_cell(anomalies_, map, frame_previous[lhs]);
                anomalies_ << ',';
                json_cell(anomalies_, map, agent.position);
                anomalies_ << "]}\n";
                ++anomaly_count_;
            }
        }
    }

    first_frame_ = false;
    steps_.flush();
    agents_.flush();
    intersections_.flush();
    schedules_.flush();
    routes_.flush();
    events_.flush();
    anomalies_.flush();
}

void DebugTrace::finish(const Simulator& simulator, const std::string_view status,
                        const double elapsed_seconds, const std::uint64_t vertex_conflicts,
                        const std::uint64_t edge_conflicts) {
    events_ << "{\"timestep\":" << simulator.stats_.timestep << ",\"type\":\"run_finished\",\"status\":";
    json_string(events_, status);
    events_ << "}\n";
    events_.flush();

    std::ofstream output = open_output(directory_ / "summary.json");
    output << "{\"status\":";
    json_string(output, status);
    output << ",\"steps\":" << simulator.stats_.timestep
           << ",\"completed\":" << simulator.stats_.completed
           << ",\"tasks_completed\":" << simulator.stats_.completed_tasks
           << ",\"agents\":" << simulator.agents_.size()
           << ",\"moves\":" << simulator.stats_.committed_moves
           << ",\"waits\":" << simulator.stats_.waits
           << ",\"deadlocks\":" << simulator.stats_.detected_deadlocks
           << ",\"elapsed_seconds\":" << elapsed_seconds
           << ",\"vertex_conflicts\":" << vertex_conflicts
           << ",\"edge_conflicts\":" << edge_conflicts
           << ",\"debug_anomalies\":" << anomaly_count_ << "}\n";
}

}  // namespace lima
