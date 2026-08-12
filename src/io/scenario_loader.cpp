#include "lima/io/scenario_loader.hpp"

#include "lima/core/grid_map.hpp"
#include "lima/intersection/topology.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace lima {

std::vector<Task> load_scenario(const std::filesystem::path& path, const std::size_t limit) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open scenario: " + path.string());
    std::string version_line;
    std::getline(input, version_line);

    std::vector<Task> tasks;
    int bucket, width, height, sx, sy, gx, gy;
    double distance;
    std::string map_name;
    while (tasks.size() < limit && input >> bucket >> map_name >> width >> height >> sx >> sy >> gx >> gy >> distance) {
        (void)bucket;
        (void)width;
        (void)height;
        (void)distance;
        tasks.push_back({{sx, sy}, {gx, gy}});
    }
    return tasks;
}

std::vector<CellId> make_goal_candidates(const GridMap& map) {
    // S cells are shared exits: many AMRs may target the same cell because an
    // AMR is removed as soon as it enters. G cells remain persistent goals.
    std::vector<CellId> goal_candidates = map.sink_cells().empty()
        ? map.goal_cells() : map.sink_cells();
    if (goal_candidates.empty()) {
        const IntersectionTopology topology = IntersectionTopology::build(map);
        goal_candidates.reserve(map.traversable_cells().size());
        for (const CellId id : map.traversable_cells()) {
            if (!topology.memberships(id).empty()) continue;
            const bool discharge_portal = std::any_of(
                map.neighbors(id).begin(), map.neighbors(id).end(), [&](const CellId neighbor) {
                    return !topology.memberships(neighbor).empty();
                });
            // Persistent goal occupancy must not turn a one-cell warehouse
            // gateway into a parking position. PIBT may use these cells for
            // transit, but completed AMRs park deeper in the outside area.
            if (!discharge_portal) goal_candidates.push_back(id);
        }
    }
    if (goal_candidates.empty())
        throw std::runtime_error("random mode found no valid goal cells outside intersections");
    return goal_candidates;
}

std::vector<Task> make_random_tasks(const GridMap& map, const std::size_t count, const std::uint64_t seed) {
    std::vector<CellId> goal_candidates = make_goal_candidates(map);
    const bool shared_sinks = !map.sink_cells().empty();
    if (!shared_sinks && count > goal_candidates.size())
        throw std::runtime_error("persistent AMRs require at least one unique goal cell per agent");

    std::vector<CellId> starts;
    starts.reserve(map.traversable_cells().size());
    const std::unordered_set<CellId> goals(map.goal_cells().begin(), map.goal_cells().end());
    for (const CellId id : map.traversable_cells()) {
        const Coord c = map.coord(id);
        if (c.x == 0 || c.y == 0 || c.x + 1 == map.width() || c.y + 1 == map.height()) continue;
        if (!goals.contains(id)) starts.push_back(id);
    }
    if (count > starts.size()) throw std::runtime_error("requested more agents than unique start cells");

    std::mt19937_64 rng(seed);
    std::shuffle(starts.begin(), starts.end(), rng);
    std::shuffle(goal_candidates.begin(), goal_candidates.end(), rng);
    std::vector<Task> tasks;
    tasks.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t goal_index = shared_sinks ? i % goal_candidates.size() : i;
        if (goal_candidates[goal_index] == starts[i]) {
            const auto replacement = std::find_if(
                goal_candidates.begin() + static_cast<std::ptrdiff_t>(goal_index + 1), goal_candidates.end(),
                [&](const CellId candidate) { return candidate != starts[i]; });
            if (replacement == goal_candidates.end())
                throw std::runtime_error("random mode cannot choose a unique goal different from its start");
            std::iter_swap(goal_candidates.begin() + static_cast<std::ptrdiff_t>(goal_index), replacement);
        }
        tasks.push_back({map.coord(starts[i]), map.coord(goal_candidates[goal_index])});
    }
    return tasks;
}

}  // namespace lima
