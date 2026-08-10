#pragma once

#include "lima/core/grid_map.hpp"

#include <array>
#include <vector>

namespace lima {

enum class Direction : std::uint8_t { North = 0, East = 1, South = 2, West = 3, Center = 4, None = 5 };

struct Intersection {
    IntersectionId id{};
    CellId center{kInvalidCell};
    std::array<std::vector<CellId>, 4> arms;  // center-near to far
    std::array<IntersectionId, 4> neighbors{{-1, -1, -1, -1}};

    [[nodiscard]] bool has_arm(Direction direction) const noexcept;
    [[nodiscard]] Direction direction_of(CellId cell) const noexcept;
    [[nodiscard]] std::vector<CellId> cells() const;
};

class IntersectionTopology {
public:
    static IntersectionTopology build(const GridMap& map);

    [[nodiscard]] const std::vector<Intersection>& intersections() const noexcept { return intersections_; }
    [[nodiscard]] const std::vector<IntersectionId>& memberships(CellId cell) const;

private:
    std::vector<Intersection> intersections_;
    std::vector<std::vector<IntersectionId>> cell_to_intersections_;
};

}  // namespace lima

