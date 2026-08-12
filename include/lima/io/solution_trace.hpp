#pragma once

#include "lima/core/agent.hpp"
#include "lima/core/grid_map.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace lima {

class SolutionTrace {
public:
    SolutionTrace() = default;
    SolutionTrace(const GridMap& map, std::span<const Agent> agents, std::string map_file,
                  bool validate_conflicts = false, bool lifelong = false);

    void append(std::span<const Agent> agents);
    void set_computation_time(double seconds) noexcept { computation_time_seconds_ = seconds; }
    void save(const std::filesystem::path& path, const GridMap& map, bool solved) const;
    static SolutionTrace load(const std::filesystem::path& path, const GridMap& map);

    [[nodiscard]] std::size_t agent_count() const noexcept { return starts_.size(); }
    [[nodiscard]] std::size_t frame_count() const noexcept {
        return agent_count() == 0 ? 0 : configurations_.size() / agent_count();
    }
    [[nodiscard]] std::span<const CellId> frame(std::size_t timestep) const;
    [[nodiscard]] std::span<const CellId> goal_frame(std::size_t timestep) const;
    [[nodiscard]] const std::vector<CellId>& starts() const noexcept { return starts_; }
    [[nodiscard]] const std::vector<CellId>& goals() const noexcept { return goals_; }
    [[nodiscard]] const std::string& map_file() const noexcept { return map_file_; }
    [[nodiscard]] bool solved() const noexcept { return solved_; }
    [[nodiscard]] bool lifelong() const noexcept { return lifelong_; }
    [[nodiscard]] std::uint64_t tasks_completed(std::size_t timestep) const noexcept {
        return timestep < task_counts_.size() ? task_counts_[timestep] : 0;
    }
    [[nodiscard]] bool validation_enabled() const noexcept { return validation_enabled_; }
    [[nodiscard]] std::uint64_t vertex_conflicts() const noexcept { return vertex_conflicts_; }
    [[nodiscard]] std::uint64_t edge_conflicts() const noexcept { return edge_conflicts_; }

private:
    std::string map_file_;
    std::vector<CellId> starts_;
    std::vector<CellId> goals_;
    std::vector<CellId> configurations_;
    std::vector<CellId> goal_configurations_;
    std::vector<std::uint64_t> task_counts_;
    std::vector<bool> previous_active_;
    std::vector<CellId> current_frame_;
    std::vector<AgentId> current_occupancy_;
    std::vector<AgentId> previous_occupancy_;
    double computation_time_seconds_{};
    std::uint64_t vertex_conflicts_{};
    std::uint64_t edge_conflicts_{};
    bool validation_enabled_{};
    bool solved_{};
    bool lifelong_{};
};

}  // namespace lima
