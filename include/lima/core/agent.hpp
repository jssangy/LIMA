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
    ExecutionFailure,
};

struct Agent {
    AgentId id{kNoAgent};
    CellId position{kInvalidCell};
    CellId goal{kInvalidCell};
    // Immutable task-level reference used by the completion rank. Runtime
    // schedules and detours may rewrite `route`, but never this suffix ledger.
    std::vector<CellId> reference_route;
    std::size_t reference_cursor{};
    std::vector<CellId> route;
    std::size_t route_cursor{};
    std::size_t scheduling_remaining{};
    std::int32_t schedule_group{kNoGroup};
    std::uint32_t wait_steps{};
    WaitReason wait_reason{WaitReason::None};
    std::uint64_t moves{};
    std::uint64_t tasks_completed{};  // lifelong mode: goals served so far
    bool active{true};
    bool reached{false};  // stay-at-goal mode: first arrival already counted
    bool awaiting_goal{false};  // lifelong mode: task served, next goal pending

    [[nodiscard]] CellId intended_cell() const noexcept {
        return route_cursor + 1 < route.size() ? route[route_cursor + 1] : position;
    }

    [[nodiscard]] bool scheduled() const noexcept { return scheduling_remaining > 0; }
};

}  // namespace lima
