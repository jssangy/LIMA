#pragma once

#include "lima/core/grid_map.hpp"

#include <memory>
#include <deque>
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
    explicit BfsPlanner(const GridMap& map) : map_(map) {}
    [[nodiscard]] std::vector<CellId> plan(CellId start, CellId goal) override;

private:
    const GridMap& map_;
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

std::unique_ptr<Planner> make_planner(PlannerKind kind, const GridMap& map);

// Local recovery route. A non-zero blocked entry excludes that cell, except start and goal.
std::vector<CellId> plan_avoiding(const GridMap& map, CellId start, CellId goal,
                                  std::span<const std::uint8_t> blocked);

}  // namespace lima
