#pragma once

#include "lima/core/agent.hpp"
#include "lima/core/grid_map.hpp"
#include "lima/intersection/coordinator.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace lima {

// Passive experiment instrumentation (revision work item W1).  Every hook only
// observes; enabling metrics must never change simulation behavior.
class MetricsCollector {
public:
    MetricsCollector(const std::filesystem::path& directory, const GridMap& map,
                     std::span<const Agent> agents);

    void on_solver_invocation(std::uint64_t timestep, IntersectionId intersection,
                              const ScheduleTelemetry& telemetry, bool accepted);
    void on_discharge(std::uint64_t timestep, IntersectionId intersection,
                      std::span<const AgentId> agents, std::span<const std::size_t> loop_cells,
                      std::span<const std::uint8_t> loop_closed);
    void on_task_completion(std::uint64_t timestep, AgentId agent,
                            std::uint64_t task_index, std::uint64_t service_steps);
    void on_acquisition(std::uint64_t timestep, AgentId agent, IntersectionId intersection,
                        int distance_cells);
    void on_broadcast(std::uint64_t timestep, IntersectionId intersection, AgentId agent,
                      int distance_cells);
    void on_gate_signal(std::uint64_t timestep, IntersectionId source, IntersectionId target,
                        int distance_cells);
    void on_route_mutation(std::uint64_t timestep, AgentId agent, std::string_view kind,
                           std::size_t inserted_cells, CellId rejoin, bool rejoin_ok,
                           bool goal_preserved);
    void observe_step(std::uint64_t timestep, std::span<const Agent> agents);
    void flush_step(std::uint64_t timestep);

    void note_discharged_agent(AgentId agent);
    void finalize(std::span<const Agent> agents,
                  std::span<const std::size_t> initial_route_lengths,
                  std::span<const std::uint64_t> completion_steps);

private:
    std::ofstream solver_csv_;
    std::ofstream discharge_csv_;
    std::ofstream comm_csv_;
    std::ofstream comm_events_csv_;
    std::ofstream agents_csv_;
    std::ofstream task_csv_;
    std::ofstream recirculation_csv_;
    std::ofstream route_mutations_csv_;
    std::ofstream path_validation_csv_;
    const GridMap& map_;
    std::vector<std::uint32_t> discharge_counts_;
    std::vector<CellId> previous_positions_;
    std::vector<std::uint8_t> previous_active_;
    std::vector<AgentId> current_occupancy_;
    std::vector<AgentId> previous_occupancy_;
    std::uint64_t observed_steps_{};
    std::uint64_t invalid_moves_{};
    std::uint64_t vertex_conflicts_{};
    std::uint64_t edge_conflicts_{};
    std::uint64_t completed_goal_mismatches_{};
    int step_acquisitions_{};
    int step_broadcasts_{};
    int step_gate_signals_{};
};

}  // namespace lima
