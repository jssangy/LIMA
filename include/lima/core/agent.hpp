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
    std::size_t scheduling_remaining{};
    std::int32_t schedule_group{kNoGroup};
    std::uint32_t wait_steps{};
    WaitReason wait_reason{WaitReason::None};
    std::uint64_t moves{};
    std::uint64_t tasks_completed{};
    std::uint64_t task_started_timestep{};
    bool completed{false};
    bool active{true};
    bool awaiting_goal{false};

    [[nodiscard]] CellId intended_cell() const noexcept {
        // A robot displaced from its completed goal by PIBT keeps the goal as
        // its local target and returns after yielding to passing traffic.
        return route_cursor + 1 < route.size() ? route[route_cursor + 1] : goal;
    }

    [[nodiscard]] bool scheduled() const noexcept { return scheduling_remaining > 0; }
};

}  // namespace lima
