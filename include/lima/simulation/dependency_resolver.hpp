#pragma once

#include "lima/core/agent.hpp"
#include "lima/core/grid_map.hpp"

#include <span>
#include <vector>

namespace lima {

struct MoveIntent {
    AgentId agent{kNoAgent};
    CellId from{kInvalidCell};
    CellId to{kInvalidCell};
    std::int32_t schedule_group{kNoGroup};
    std::uint32_t wait_steps{};
};

struct MoveResolution {
    std::vector<bool> approved;
    std::vector<WaitReason> reasons;
};

class DependencyResolver {
public:
    explicit DependencyResolver(const GridMap& map) : map_(map) {}

    // Result indices match the intent span. Scheduled groups are all-or-nothing.
    [[nodiscard]] MoveResolution resolve(std::span<const MoveIntent> intents) const;
    void validate_transition(std::span<const MoveIntent> intents, const std::vector<bool>& approved) const;

private:
    const GridMap& map_;
};

}  // namespace lima
