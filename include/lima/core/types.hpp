#pragma once

#include <cstdint>
#include <limits>

namespace lima {

using CellId = std::int32_t;
using AgentId = std::int32_t;
using IntersectionId = std::int32_t;

inline constexpr CellId kInvalidCell = -1;
inline constexpr AgentId kNoAgent = -1;
inline constexpr std::int32_t kNoGroup = -1;

struct Coord {
    int x{};
    int y{};

    friend constexpr bool operator==(Coord, Coord) = default;
};

struct Task {
    Coord start;
    Coord goal;
};

enum class PlannerKind : std::uint8_t { Bfs, AStar };

}  // namespace lima

