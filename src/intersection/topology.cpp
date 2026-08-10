#include "lima/intersection/topology.hpp"

#include <algorithm>
#include <unordered_map>

namespace lima {
namespace {

constexpr std::array<Coord, 4> kDelta{{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}};

bool wall_or_outside(const GridMap& map, const Coord c) { return !map.in_bounds(c) || !map.traversable(c); }

bool is_center(const GridMap& map, const Coord c) {
    if (!map.traversable(c)) return false;
    int open = 0;
    for (const Coord d : kDelta) open += map.traversable({c.x + d.x, c.y + d.y}) ? 1 : 0;
    if (open < 3) return false;
    const std::array<Coord, 4> diagonal{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};
    return std::all_of(diagonal.begin(), diagonal.end(), [&](const Coord d) {
        return wall_or_outside(map, {c.x + d.x, c.y + d.y});
    });
}

std::vector<CellId> trace_arm(const GridMap& map, const Coord center, const Direction direction) {
    const Coord delta = kDelta[static_cast<std::size_t>(direction)];
    std::vector<CellId> cells;
    Coord c{center.x + delta.x, center.y + delta.y};
    while (map.traversable(c)) {
        if (std::find(map.goal_cells().begin(), map.goal_cells().end(), map.cell(c)) != map.goal_cells().end()) break;
        const bool corridor = delta.y != 0
            ? wall_or_outside(map, {c.x - 1, c.y}) && wall_or_outside(map, {c.x + 1, c.y})
            : wall_or_outside(map, {c.x, c.y - 1}) && wall_or_outside(map, {c.x, c.y + 1});
        if (!corridor) break;
        cells.push_back(map.cell(c));
        c = {c.x + delta.x, c.y + delta.y};
    }
    return cells;
}

}  // namespace

bool Intersection::has_arm(const Direction direction) const noexcept {
    const auto value = static_cast<std::size_t>(direction);
    return value < 4 && !arms[value].empty();
}

Direction Intersection::direction_of(const CellId cell) const noexcept {
    if (cell == center) return Direction::Center;
    for (std::size_t d = 0; d < arms.size(); ++d) {
        if (std::find(arms[d].begin(), arms[d].end(), cell) != arms[d].end()) return static_cast<Direction>(d);
    }
    return Direction::None;
}

std::vector<CellId> Intersection::cells() const {
    std::vector<CellId> result{center};
    for (const auto& arm : arms) result.insert(result.end(), arm.begin(), arm.end());
    return result;
}

IntersectionTopology IntersectionTopology::build(const GridMap& map) {
    IntersectionTopology topology;
    topology.cell_to_intersections_.resize(static_cast<std::size_t>(map.cell_count()));
    std::unordered_map<CellId, IntersectionId> by_center;

    for (const CellId cell : map.traversable_cells()) {
        const Coord center = map.coord(cell);
        if (!is_center(map, center)) continue;
        Intersection intersection;
        intersection.id = static_cast<IntersectionId>(topology.intersections_.size());
        intersection.center = cell;
        int arms = 0;
        for (std::size_t d = 0; d < 4; ++d) {
            intersection.arms[d] = trace_arm(map, center, static_cast<Direction>(d));
            arms += intersection.arms[d].empty() ? 0 : 1;
        }
        if (arms < 3) continue;
        by_center.emplace(cell, intersection.id);
        topology.intersections_.push_back(std::move(intersection));
    }

    for (auto& intersection : topology.intersections_) {
        for (std::size_t d = 0; d < 4; ++d) {
            const auto& arm = intersection.arms[d];
            if (arm.empty()) continue;
            const Coord tip = map.coord(arm.back());
            const Coord delta = kDelta[d];
            const CellId expected = map.in_bounds({tip.x + delta.x, tip.y + delta.y})
                ? map.cell({tip.x + delta.x, tip.y + delta.y}) : kInvalidCell;
            if (const auto it = by_center.find(expected); it != by_center.end()) intersection.neighbors[d] = it->second;
        }
        for (const CellId cell : intersection.cells()) {
            topology.cell_to_intersections_[static_cast<std::size_t>(cell)].push_back(intersection.id);
        }
    }
    return topology;
}

const std::vector<IntersectionId>& IntersectionTopology::memberships(const CellId cell) const {
    static const std::vector<IntersectionId> empty;
    if (cell < 0 || static_cast<std::size_t>(cell) >= cell_to_intersections_.size()) return empty;
    return cell_to_intersections_[static_cast<std::size_t>(cell)];
}

}  // namespace lima
