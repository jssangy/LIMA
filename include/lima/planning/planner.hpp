#pragma once

#include "lima/core/grid_map.hpp"

#include <memory>
#include <deque>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>

namespace lima {

class Planner {
public:
    virtual ~Planner() = default;
    [[nodiscard]] virtual std::vector<CellId> plan(CellId start, CellId goal) = 0;
};

class BfsPlanner final : public Planner {
public:
    BfsPlanner(const GridMap& map, std::mt19937_64& rng) : map_(map), rng_(rng) {}
    [[nodiscard]] std::vector<CellId> plan(CellId start, CellId goal) override;

private:
    const GridMap& map_;
    std::mt19937_64& rng_;
    std::unordered_map<CellId, std::vector<std::int32_t>> fields_;
    std::deque<CellId> field_order_;
};

class AStarPlanner final : public Planner {
public:
    explicit AStarPlanner(const GridMap& map) : map_(map) {}
    [[nodiscard]] std::vector<CellId> plan(CellId start, CellId goal) override;

private:
    const GridMap& map_;
};

std::unique_ptr<Planner> make_planner(PlannerKind kind, const GridMap& map, std::mt19937_64& rng);

struct SuffixRepair {
    // Executable route beginning at the agent's current cell and ending at the
    // unchanged reference goal.
    std::vector<CellId> route;
    CellId rejoin{kInvalidCell};
    std::size_t reference_rejoin_index{};
    std::size_t bridge_edges{};
};

// Reconnect a displaced agent to the first unfinished reference waypoint and
// preserve the entire suffix after it. This is the Route Planner's recovery
// contract; local coordination never replaces an unfinished reference suffix.
std::optional<SuffixRepair> repair_to_reference_suffix(
    Planner& planner, CellId current, std::span<const CellId> reference_route,
    std::size_t reference_cursor);

// Local recovery route. A non-zero blocked entry excludes that cell, except start and goal.
std::vector<CellId> plan_avoiding(const GridMap& map, CellId start, CellId goal,
                                  std::span<const std::uint8_t> blocked);

}  // namespace lima
