#pragma once

#include "lima/core/types.hpp"

#include <cstddef>
#include <vector>

namespace lima {

enum class WaitReason : std::uint8_t {
    None,
    ScheduledHold,
    IntersectionReserved,
    IntersectionCapacity,
    VertexConflict,
    EdgeSwap,
    Dependency,
    ScheduleGroup,
};

struct Agent {
    AgentId id{kNoAgent};
    CellId position{kInvalidCell};
    CellId goal{kInvalidCell};
    std::vector<CellId> route;
    std::size_t route_cursor{};
    std::vector<CellId> schedule_route;
    std::size_t schedule_cursor{};
    std::int32_t schedule_group{kNoGroup};
    // A trimmed schedule may finish at an arm tip before the robot crosses the
    // reserved downstream boundary.  Keep that reservation ownership separate
    // from schedule-route execution.
    std::int32_t discharge_group{kNoGroup};
    std::uint32_t wait_steps{};
    WaitReason wait_reason{WaitReason::None};
    std::uint64_t moves{};
    bool active{true};

    [[nodiscard]] CellId intended_cell() const noexcept {
        if (schedule_cursor + 1 < schedule_route.size()) return schedule_route[schedule_cursor + 1];
        return route_cursor + 1 < route.size() ? route[route_cursor + 1] : position;
    }

    [[nodiscard]] bool scheduled() const noexcept { return schedule_group != kNoGroup; }
};

}  // namespace lima
