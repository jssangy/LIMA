#include "lima/simulation/metrics.hpp"

#include <stdexcept>

namespace lima {

MetricsCollector::MetricsCollector(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const auto open = [&](std::ofstream& stream, const char* name, const char* header) {
        stream.open(directory / name, std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot open metrics file: " + (directory / name).string());
        stream << header << '\n';
    };
    open(solver_csv_, "solver_invocations.csv",
         "t,intersection,intents,outcome,accepted,iterations,expanded,wall_us,solution_len,fastpath");
    open(discharge_csv_, "discharge_events.csv", "t,intersection,rerouted,loop_cells,agents");
    open(comm_csv_, "comm_steps.csv", "t,acquisitions,broadcasts,gate_signals");
    open(agents_csv_, "agents.csv",
         "id,initial_route_len,moves,extra_moves,completion_step,discharges,completed");
}

void MetricsCollector::on_solver_invocation(const std::uint64_t timestep, const IntersectionId intersection,
                                            const ScheduleTelemetry& telemetry, const bool accepted) {
    solver_csv_ << timestep << ',' << intersection << ',' << telemetry.intents << ','
                << telemetry.solver.outcome << ',' << (accepted ? 1 : 0) << ','
                << telemetry.solver.iterations << ',' << telemetry.solver.expanded_nodes << ','
                << static_cast<std::uint64_t>(telemetry.solver.wall_seconds * 1e6) << ','
                << telemetry.solver.solution_length << ',' << (telemetry.solver.fastpath_solved ? 1 : 0) << '\n';
}

void MetricsCollector::on_discharge(const std::uint64_t timestep, const IntersectionId intersection,
                                    const std::span<const AgentId> agents, const std::size_t loop_cells) {
    discharge_csv_ << timestep << ',' << intersection << ',' << agents.size() << ',' << loop_cells << ',';
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) discharge_csv_ << ' ';
        discharge_csv_ << agents[i];
    }
    discharge_csv_ << '\n';
}

void MetricsCollector::flush_step(const std::uint64_t timestep) {
    if (step_acquisitions_ != 0 || step_broadcasts_ != 0 || step_gate_signals_ != 0) {
        comm_csv_ << timestep << ',' << step_acquisitions_ << ',' << step_broadcasts_ << ','
                  << step_gate_signals_ << '\n';
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
                    << completion << ',' << discharges << ',' << (agent.active ? 0 : 1) << '\n';
    }
    solver_csv_.flush();
    discharge_csv_.flush();
    comm_csv_.flush();
    agents_csv_.flush();
}

}  // namespace lima
