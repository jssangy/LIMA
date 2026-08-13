#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/deadlock_detector.hpp"
#include "lima/scheduling/solver.hpp"

#include <cstdint>
#include <array>
#include <optional>
#include <span>
#include <vector>

namespace lima {

struct ScheduledPath {
    AgentId agent{kNoAgent};
    std::vector<CellId> path;
    CellId target_exit{kInvalidCell};
};

// Telemetry for one scheduling attempt, consumed by the metrics layer.
struct ScheduleTelemetry {
    std::size_t intents{};
    SolverStats solver;
};

class IntersectionCoordinator {
public:
    // strict_release enables the PIBT-corridor companion rules: egress AMRs
    // are additionally kept under schedule while any scheduled path moves
    // inward on their arm, and trimmed egress hands off only at the arm tip.
    // Both rules rely on the PIBT movement phase to evacuate the retained
    // cohort afterwards; without PIBT they livelock dense cells, so they stay
    // off in the default configuration.
    explicit IntersectionCoordinator(StackSolver& solver, const bool strict_release = false)
        : solver_(solver), strict_release_(strict_release) {}

    // arm_limits caps the usable depth of each arm (cells beyond the limit are
    // treated as walls); pass the arm sizes for the classic full-arm behavior.
    std::optional<std::vector<ScheduledPath>> schedule(
        const Intersection& intersection, std::span<const IntersectionIntent> intents,
        const std::array<int, 4>& stack_quotas, const std::array<int, 4>& arm_limits,
        ScheduleTelemetry* telemetry = nullptr) const;

private:
    StackSolver& solver_;
    bool strict_release_{false};
};

}  // namespace lima
