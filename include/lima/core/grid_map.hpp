#pragma once

#include "lima/core/types.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace lima {

class GridMap {
public:
    static GridMap load(const std::filesystem::path& path);

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] int cell_count() const noexcept { return width_ * height_; }
    [[nodiscard]] bool in_bounds(Coord c) const noexcept;
    [[nodiscard]] bool traversable(Coord c) const noexcept;
    [[nodiscard]] bool traversable(CellId id) const noexcept;
    [[nodiscard]] bool goal(CellId id) const noexcept;
    [[nodiscard]] bool terminal(CellId id) const noexcept;
    [[nodiscard]] CellId cell(Coord c) const noexcept;
    [[nodiscard]] Coord coord(CellId id) const noexcept;
    [[nodiscard]] const std::vector<CellId>& neighbors(CellId id) const;
    [[nodiscard]] const std::vector<CellId>& goal_cells() const noexcept { return goals_; }
    [[nodiscard]] const std::vector<CellId>& sink_cells() const noexcept { return sinks_; }
    [[nodiscard]] const std::vector<CellId>& traversable_cells() const noexcept { return free_cells_; }

private:
    int width_{};
    int height_{};
    std::vector<std::uint8_t> blocked_;
    std::vector<std::uint8_t> goal_mask_;
    std::vector<std::uint8_t> terminal_mask_;
    std::vector<std::vector<CellId>> neighbors_;
    std::vector<CellId> goals_;
    std::vector<CellId> sinks_;
    std::vector<CellId> free_cells_;
};

}  // namespace lima
