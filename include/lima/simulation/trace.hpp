#pragma once

#include "lima/core/agent.hpp"
#include "lima/core/grid_map.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace lima {

// JSONL step tracer for the AI-agent debug harness.  One header record with
// the instance (map, tasks, initial routes) followed by one record per
// timestep carrying every agent position plus the structured events of that
// step.  tools/verify_trace.py replays the file and checks the simulation
// invariants offline.
class StepTracer {
public:
    struct GroupEvent {
        IntersectionId intersection{-1};
        std::vector<AgentId> agents;
    };

    StepTracer(const std::filesystem::path& path, const GridMap& map, std::string map_file,
               std::uint64_t seed, std::span<const Agent> agents);

    void add_schedule(IntersectionId intersection, std::vector<AgentId> agents);
    void add_discharge(IntersectionId intersection, std::vector<AgentId> agents);
    void add_completion(AgentId agent);
    void flush_step(std::uint64_t timestep, std::span<const Agent> agents);
    void finish();  // flush and fail loudly on write errors

private:
    std::ofstream out_;
    std::vector<GroupEvent> schedules_;
    std::vector<GroupEvent> discharges_;
    std::vector<AgentId> completions_;
};

}  // namespace lima
