#pragma once

#include "lima/core/types.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace lima {

std::vector<Task> load_scenario(const std::filesystem::path& path, std::size_t limit);
std::vector<CellId> make_goal_candidates(const class GridMap& map);
std::vector<Task> make_random_tasks(const class GridMap& map, std::size_t count, std::uint64_t seed);

}  // namespace lima
