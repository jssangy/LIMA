#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/coordinator.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace lima {

// Passive experiment instrumentation (revision work item W1).  Every hook only
// observes; enabling metrics must never change simulation behavior.
class MetricsCollector {
public:
    explicit MetricsCollector(const std::filesystem::path& directory);

    void on_solver_invocation(std::uint64_t timestep, IntersectionId intersection,
                              const ScheduleTelemetry& telemetry, bool accepted);
    void on_discharge(std::uint64_t timestep, IntersectionId intersection,
                      std::span<const AgentId> agents, std::size_t loop_cells);
    void on_task_completion(std::uint64_t timestep, AgentId agent,
                            std::uint64_t task_index, std::uint64_t service_steps);
    void add_acquisitions(int count) noexcept { step_acquisitions_ += count; }
    void add_broadcasts(int count) noexcept { step_broadcasts_ += count; }
    void add_gate_signals(int count) noexcept { step_gate_signals_ += count; }
    void flush_step(std::uint64_t timestep);

    void note_discharged_agent(AgentId agent);
    void finalize(std::span<const Agent> agents,
                  std::span<const std::size_t> initial_route_lengths,
                  std::span<const std::uint64_t> completion_steps);

private:
    std::ofstream solver_csv_;
    std::ofstream discharge_csv_;
    std::ofstream comm_csv_;
    std::ofstream agents_csv_;
    std::ofstream task_csv_;
    std::vector<std::uint32_t> discharge_counts_;
    int step_acquisitions_{};
    int step_broadcasts_{};
    int step_gate_signals_{};
};

}  // namespace lima
