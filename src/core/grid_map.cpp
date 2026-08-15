#include "lima/core/grid_map.hpp"

#include <fstream>
#include <stdexcept>

namespace lima {

GridMap GridMap::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open map: " + path.string());

    GridMap map;
    std::string token;
    std::string type;
    if (!(input >> token >> type) || token != "type") throw std::runtime_error("invalid map header");
    if (!(input >> token >> map.height_) || token != "height") throw std::runtime_error("missing map height");
    if (!(input >> token >> map.width_) || token != "width") throw std::runtime_error("missing map width");
    if (!(input >> token) || token != "map") throw std::runtime_error("missing map body");
    if (map.width_ <= 0 || map.height_ <= 0) throw std::runtime_error("invalid map dimensions");

    map.blocked_.assign(static_cast<std::size_t>(map.cell_count()), 1);
    map.goal_mask_.assign(static_cast<std::size_t>(map.cell_count()), 0);
    map.terminal_mask_.assign(static_cast<std::size_t>(map.cell_count()), 0);
    std::string row;
    for (int y = 0; y < map.height_; ++y) {
        if (!(input >> row) || static_cast<int>(row.size()) != map.width_) {
            throw std::runtime_error("invalid map row " + std::to_string(y));
        }
        for (int x = 0; x < map.width_; ++x) {
            const char value = row[static_cast<std::size_t>(x)];
            const bool free = value == '.' || value == 'S' || value == 'E' || value == 'G';
            if (!free && value != '@' && value != 'T') {
                throw std::runtime_error("unsupported map cell character");
            }
            const CellId id = y * map.width_ + x;
            map.blocked_[static_cast<std::size_t>(id)] = free ? 0 : 1;
            if (free) map.free_cells_.push_back(id);
            if (value == 'S' || value == 'G') {
                map.goals_.push_back(id);
                map.goal_mask_[static_cast<std::size_t>(id)] = 1;
            }
            if (value == 'S') map.sinks_.push_back(id);
            if (value == 'E') map.terminal_mask_[static_cast<std::size_t>(id)] = 1;
        }
    }

    map.neighbors_.resize(static_cast<std::size_t>(map.cell_count()));
    constexpr std::array<Coord, 4> offsets{{{0, -1}, {-1, 0}, {1, 0}, {0, 1}}};
    for (const CellId id : map.free_cells_) {
        const Coord c = map.coord(id);
        auto& out = map.neighbors_[static_cast<std::size_t>(id)];
        out.reserve(4);
        for (const Coord d : offsets) {
            const Coord next{c.x + d.x, c.y + d.y};
            if (map.traversable(next)) out.push_back(map.cell(next));
        }
    }
    return map;
}

bool GridMap::in_bounds(const Coord c) const noexcept {
    return c.x >= 0 && c.y >= 0 && c.x < width_ && c.y < height_;
}

bool GridMap::traversable(const Coord c) const noexcept {
    return in_bounds(c) && blocked_[static_cast<std::size_t>(cell(c))] == 0;
}

bool GridMap::traversable(const CellId id) const noexcept {
    return id >= 0 && id < cell_count() && blocked_[static_cast<std::size_t>(id)] == 0;
}

bool GridMap::goal(const CellId id) const noexcept {
    return id >= 0 && id < cell_count() && goal_mask_[static_cast<std::size_t>(id)] != 0;
}

bool GridMap::terminal(const CellId id) const noexcept {
    return id >= 0 && id < cell_count() && terminal_mask_[static_cast<std::size_t>(id)] != 0;
}

CellId GridMap::cell(const Coord c) const noexcept { return c.y * width_ + c.x; }

Coord GridMap::coord(const CellId id) const noexcept { return {id % width_, id / width_}; }

const std::vector<CellId>& GridMap::neighbors(const CellId id) const {
    if (id < 0 || id >= cell_count()) throw std::out_of_range("invalid cell id");
    return neighbors_[static_cast<std::size_t>(id)];
}

}  // namespace lima
