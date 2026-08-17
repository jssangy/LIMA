#include "lima/simulation/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lima {

MetricsCollector::MetricsCollector(const std::filesystem::path& directory, const GridMap& map,
                                   const std::span<const Agent> agents)
    : map_(map) {
    std::filesystem::create_directories(directory);
    const auto open = [&](std::ofstream& stream, const char* name, const char* header) {
        stream.open(directory / name, std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot open metrics file: " + (directory / name).string());
        stream << header << '\n';
    };
    open(solver_csv_, "solver_invocations.csv",
         "t,intersection,intents,outcome,accepted,iterations,expanded,wall_us,solution_len,fastpath,fallback");
    open(discharge_csv_, "discharge_events.csv", "t,intersection,rerouted,loop_cells,agents");
    open(comm_csv_, "comm_steps.csv", "t,acquisitions,broadcasts,gate_signals");
    open(comm_events_csv_, "communication_events.csv",
         "t,type,source,target,distance_cells,intersection_hops");
    open(agents_csv_, "agents.csv",
         "id,initial_route_len,moves,extra_moves,completion_step,discharges,completed,tasks_completed");
    open(task_csv_, "task_completions.csv", "t,agent,task_index,service_steps");
    open(recirculation_csv_, "recirculation_segments.csv",
         "t,intersection,agent,loop_cells,closed");
    open(route_mutations_csv_, "route_mutations.csv",
         "t,agent,type,inserted_cells,rejoin,rejoin_ok,goal_preserved");
    open(path_validation_csv_, "path_validation.csv",
         "steps_observed,invalid_moves,vertex_conflicts,edge_conflicts,completed_goal_mismatches,ok");
    open(controller_information_csv_, "controller_information_events.csv",
         "t,mechanism,event,source,target,count,bytes,hops,delay_steps");
    previous_positions_.reserve(agents.size());
    previous_active_.reserve(agents.size());
    for (const Agent& agent : agents) {
        previous_positions_.push_back(agent.position);
        previous_active_.push_back(agent.active ? 1 : 0);
    }
    current_occupancy_.resize(static_cast<std::size_t>(map_.cell_count()), kNoAgent);
    previous_occupancy_.resize(static_cast<std::size_t>(map_.cell_count()), kNoAgent);
}

void MetricsCollector::on_solver_invocation(const std::uint64_t timestep, const IntersectionId intersection,
                                            const ScheduleTelemetry& telemetry, const bool accepted) {
    solver_csv_ << timestep << ',' << intersection << ',' << telemetry.intents << ','
                << telemetry.solver.outcome << ',' << (accepted ? 1 : 0) << ','
                << telemetry.solver.iterations << ',' << telemetry.solver.expanded_nodes << ','
                << static_cast<std::uint64_t>(telemetry.solver.wall_seconds * 1e6) << ','
                << telemetry.solver.solution_length << ',' << (telemetry.solver.fastpath_solved ? 1 : 0) << ','
                << (telemetry.solver.fallback_used ? 1 : 0) << '\n';
}

void MetricsCollector::on_discharge(const std::uint64_t timestep, const IntersectionId intersection,
                                    const std::span<const AgentId> agents,
                                    const std::span<const std::size_t> loop_cells,
                                    const std::span<const std::uint8_t> loop_closed) {
    const std::size_t maximum = loop_cells.empty()
        ? 0 : *std::max_element(loop_cells.begin(), loop_cells.end());
    discharge_csv_ << timestep << ',' << intersection << ',' << agents.size() << ',' << maximum << ',';
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) discharge_csv_ << ' ';
        discharge_csv_ << agents[i];
    }
    discharge_csv_ << '\n';
    for (std::size_t i = 0; i < agents.size(); ++i) {
        const std::size_t length = i < loop_cells.size() ? loop_cells[i] : 0;
        const bool closed = i < loop_closed.size() && loop_closed[i] != 0;
        recirculation_csv_ << timestep << ',' << intersection << ',' << agents[i] << ','
                           << length << ',' << (closed ? 1 : 0) << '\n';
    }
}

void MetricsCollector::on_acquisition(const std::uint64_t timestep, const AgentId agent,
                                      const IntersectionId intersection, const int distance_cells) {
    ++step_acquisitions_;
    comm_events_csv_ << timestep << ",acquisition," << agent << ',' << intersection << ','
                     << distance_cells << ",0\n";
}

void MetricsCollector::on_broadcast(const std::uint64_t timestep, const IntersectionId intersection,
                                    const AgentId agent, const int distance_cells) {
    ++step_broadcasts_;
    comm_events_csv_ << timestep << ",broadcast," << intersection << ',' << agent << ','
                     << distance_cells << ",0\n";
}

void MetricsCollector::on_gate_signal(const std::uint64_t timestep, const IntersectionId source,
                                      const IntersectionId target, const int distance_cells) {
    ++step_gate_signals_;
    comm_events_csv_ << timestep << ",gate_signal," << source << ',' << target << ','
                     << distance_cells << ",1\n";
}

void MetricsCollector::on_route_mutation(const std::uint64_t timestep, const AgentId agent,
                                         const std::string_view kind,
                                         const std::size_t inserted_cells, const CellId rejoin,
                                         const bool rejoin_ok, const bool goal_preserved) {
    route_mutations_csv_ << timestep << ',' << agent << ',' << kind << ',' << inserted_cells << ','
                         << rejoin << ',' << (rejoin_ok ? 1 : 0) << ','
                         << (goal_preserved ? 1 : 0) << '\n';
}

void MetricsCollector::on_controller_information(
    const std::uint64_t timestep, const std::string_view mechanism,
    const std::string_view event, const IntersectionId source,
    const IntersectionId target, const std::uint64_t count,
    const std::uint32_t bytes, const std::uint16_t hops,
    const std::uint16_t delay_steps) {
    controller_information_csv_ << timestep << ',' << mechanism << ',' << event << ','
                                << source << ',' << target << ',' << count << ',' << bytes
                                << ',' << hops << ',' << delay_steps << '\n';
}

void MetricsCollector::observe_step(const std::uint64_t, const std::span<const Agent> agents) {
    if (agents.size() != previous_positions_.size())
        throw std::logic_error("metrics agent count changed");
    ++observed_steps_;
    std::fill(current_occupancy_.begin(), current_occupancy_.end(), kNoAgent);
    std::fill(previous_occupancy_.begin(), previous_occupancy_.end(), kNoAgent);
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (previous_active_[i])
            previous_occupancy_[static_cast<std::size_t>(previous_positions_[i])] =
                static_cast<AgentId>(i);
        const bool participates = previous_active_[i] || agents[i].active;
        if (!participates) continue;
        AgentId& occupant = current_occupancy_[static_cast<std::size_t>(agents[i].position)];
        if (occupant != kNoAgent) ++vertex_conflicts_;
        else occupant = static_cast<AgentId>(i);
        const Coord before = map_.coord(previous_positions_[i]);
        const Coord after = map_.coord(agents[i].position);
        const int distance = std::abs(before.x - after.x) + std::abs(before.y - after.y);
        if (previous_active_[i] && distance > 1) ++invalid_moves_;
        if (previous_active_[i] && !agents[i].active && agents[i].position != agents[i].goal)
            ++completed_goal_mismatches_;
    }
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (!previous_active_[i] || previous_positions_[i] == agents[i].position) continue;
        const AgentId occupant = previous_occupancy_[static_cast<std::size_t>(agents[i].position)];
        if (occupant == kNoAgent) continue;
        const std::size_t j = static_cast<std::size_t>(occupant);
        if (j > i && previous_active_[j] && previous_positions_[j] == agents[i].position
            && agents[j].position == previous_positions_[i]) ++edge_conflicts_;
    }
    previous_positions_.clear();
    previous_active_.clear();
    for (const Agent& agent : agents) {
        previous_positions_.push_back(agent.position);
        previous_active_.push_back(agent.active ? 1 : 0);
    }
}

void MetricsCollector::on_task_completion(const std::uint64_t timestep, const AgentId agent,
                                          const std::uint64_t task_index,
                                          const std::uint64_t service_steps) {
    task_csv_ << timestep << ',' << agent << ',' << task_index << ',' << service_steps << '\n';
}

void MetricsCollector::flush_step(const std::uint64_t timestep) {
    // One row per simulated step (including all-zero steps) so per-step
    // statistics have their denominator in-band.
    comm_csv_ << timestep << ',' << step_acquisitions_ << ',' << step_broadcasts_ << ','
              << step_gate_signals_ << '\n';
    // Preserve bounded progress when a long experiment is terminated by its
    // wall-clock guard. Flushing every 1024 rows keeps the overhead negligible.
    if ((timestep & 1023U) == 0U) {
        solver_csv_.flush();
        discharge_csv_.flush();
        comm_csv_.flush();
        comm_events_csv_.flush();
        task_csv_.flush();
        recirculation_csv_.flush();
        route_mutations_csv_.flush();
        controller_information_csv_.flush();
    }
    step_acquisitions_ = 0;
    step_broadcasts_ = 0;
    step_gate_signals_ = 0;
}

void MetricsCollector::note_discharged_agent(const AgentId agent) {
    if (agent < 0) return;
    if (static_cast<std::size_t>(agent) >= discharge_counts_.size()) {
        discharge_counts_.resize(static_cast<std::size_t>(agent) + 1, 0);
    }
    ++discharge_counts_[static_cast<std::size_t>(agent)];
}

void MetricsCollector::finalize(const std::span<const Agent> agents,
                                const std::span<const std::size_t> initial_route_lengths,
                                const std::span<const std::uint64_t> completion_steps) {
    for (std::size_t i = 0; i < agents.size(); ++i) {
        const Agent& agent = agents[i];
        const std::size_t initial = i < initial_route_lengths.size() ? initial_route_lengths[i] : 0;
        const std::int64_t shortest = initial > 0 ? static_cast<std::int64_t>(initial) - 1 : 0;
        const std::int64_t extra = static_cast<std::int64_t>(agent.moves) - shortest;
        const std::uint64_t completion = i < completion_steps.size() ? completion_steps[i] : 0;
        const std::uint32_t discharges = static_cast<std::size_t>(agent.id) < discharge_counts_.size()
            ? discharge_counts_[static_cast<std::size_t>(agent.id)] : 0;
        agents_csv_ << agent.id << ',' << initial << ',' << agent.moves << ',' << extra << ','
                    << completion << ',' << discharges << ',' << (agent.active ? 0 : 1) << ','
                    << agent.tasks_completed << '\n';
    }
    solver_csv_.flush();
    discharge_csv_.flush();
    comm_csv_.flush();
    agents_csv_.flush();
    task_csv_.flush();
    path_validation_csv_ << observed_steps_ << ',' << invalid_moves_ << ','
                         << vertex_conflicts_ << ',' << edge_conflicts_ << ','
                         << completed_goal_mismatches_ << ','
                         << (invalid_moves_ == 0 && vertex_conflicts_ == 0 && edge_conflicts_ == 0
                             && completed_goal_mismatches_ == 0 ? 1 : 0) << '\n';
    comm_events_csv_.flush();
    recirculation_csv_.flush();
    route_mutations_csv_.flush();
    path_validation_csv_.flush();
    controller_information_csv_.flush();
    for (const std::ofstream* stream : {
             &solver_csv_, &discharge_csv_, &comm_csv_, &comm_events_csv_, &agents_csv_,
             &task_csv_, &recirculation_csv_, &route_mutations_csv_, &path_validation_csv_,
             &controller_information_csv_}) {
        if (!stream->good()) throw std::runtime_error("metrics write failed (disk full?)");
    }
}

}  // namespace lima
