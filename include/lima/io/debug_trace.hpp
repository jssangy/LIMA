#pragma once

#include "lima/core/types.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lima {

class GridMap;
class Simulator;

class DebugTrace {
public:
    DebugTrace(const std::filesystem::path& directory, const Simulator& simulator,
               std::string map_file, std::string scenario_file, std::string command_line,
               std::uint64_t seed, PlannerKind planner, std::uint64_t max_steps);

    void append(const Simulator& simulator, std::string_view phase = "step");
    void finish(const Simulator& simulator, std::string_view status, double elapsed_seconds,
                std::uint64_t vertex_conflicts, std::uint64_t edge_conflicts);

    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }

private:
    std::filesystem::path directory_;
    std::ofstream steps_;
    std::ofstream agents_;
    std::ofstream intersections_;
    std::ofstream schedules_;
    std::ofstream routes_;
    std::ofstream events_;
    std::ofstream anomalies_;
    std::vector<CellId> previous_positions_;
    std::vector<CellId> previous_goals_;
    std::vector<std::size_t> previous_route_hashes_;
    std::vector<std::size_t> previous_scheduling_;
    std::vector<std::int32_t> previous_schedule_groups_;
    std::vector<std::uint32_t> previous_wait_steps_;
    std::vector<std::uint8_t> previous_completed_;
    std::vector<std::uint8_t> previous_intersection_active_;
    std::vector<std::uint8_t> previous_intersection_waiting_;
    std::uint64_t anomaly_count_{};
    bool first_frame_{true};

    void write_metadata(const Simulator& simulator, std::string_view map_file,
                        std::string_view scenario_file, std::string_view command_line,
                        std::uint64_t seed, PlannerKind planner, std::uint64_t max_steps);
    void write_schema() const;
};

}  // namespace lima
