#include "lima/io/scenario_loader.hpp"

#include "lima/core/grid_map.hpp"

#include <algorithm>
#include <fstream>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

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

std::vector<Task> make_random_tasks(const GridMap& map, const std::size_t count, const std::uint64_t seed) {
    if (map.goal_cells().empty()) throw std::runtime_error("random mode requires S or G cells in the map");

    // Benchmark maps may contain traversable pockets that are walled off from
    // the workstations.  Restrict start sampling to cells that can actually
    // reach a goal, otherwise routing fails for agents spawned in a pocket.
    std::vector<std::uint8_t> reachable(static_cast<std::size_t>(map.cell_count()), 0);
    std::queue<CellId> frontier;
    for (const CellId goal : map.goal_cells()) {
        if (!reachable[static_cast<std::size_t>(goal)]) {
            reachable[static_cast<std::size_t>(goal)] = 1;
            frontier.push(goal);
        }
    }
    while (!frontier.empty()) {
        const CellId current = frontier.front();
        frontier.pop();
        for (const CellId next : map.neighbors(current)) {
            if (reachable[static_cast<std::size_t>(next)]) continue;
            reachable[static_cast<std::size_t>(next)] = 1;
            frontier.push(next);
        }
    }

    std::vector<CellId> starts;
    starts.reserve(map.traversable_cells().size());
    const std::unordered_set<CellId> goals(map.goal_cells().begin(), map.goal_cells().end());
    for (const CellId id : map.traversable_cells()) {
        const Coord c = map.coord(id);
        if (c.x == 0 || c.y == 0 || c.x + 1 == map.width() || c.y + 1 == map.height()) continue;
        if (!reachable[static_cast<std::size_t>(id)]) continue;
        if (!goals.contains(id)) starts.push_back(id);
    }
    if (count > starts.size()) {
        throw std::runtime_error("requested " + std::to_string(count) + " agents but only "
                                 + std::to_string(starts.size()) + " goal-reachable start cells exist");
    }

    std::mt19937_64 rng(seed);
    std::shuffle(starts.begin(), starts.end(), rng);
    std::uniform_int_distribution<std::size_t> choose_goal(0, map.goal_cells().size() - 1);
    std::vector<Task> tasks;
    tasks.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const CellId goal = map.goal_cells()[choose_goal(rng)];
        tasks.push_back({map.coord(starts[i]), map.coord(goal)});
    }
    return tasks;
}

}  // namespace lima
