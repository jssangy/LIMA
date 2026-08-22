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
    virtual void note_assigned_route(std::span<const CellId>) {}
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

// A deterministic fleet-wide guidance graph: every undirected grid edge has
// one preferred direction fixed by the experiment seed. Traversing against
// the guide remains legal but carries a small static cost penalty.
class StaticGuidancePlanner final : public Planner {
public:
    StaticGuidancePlanner(const GridMap& map, std::uint64_t seed, double penalty = 0.20)
        : map_(map), seed_(seed), penalty_(penalty) {}
    [[nodiscard]] std::vector<CellId> plan(CellId start, CellId goal) override;

private:
    [[nodiscard]] double edge_cost(CellId source, CellId destination) const;

    const GridMap& map_;
    std::uint64_t seed_{};
    double penalty_{0.20};
    std::unordered_map<CellId, std::vector<double>> fields_;
    std::deque<CellId> field_order_;
};

// Online traffic-flow guidance. Each newly assigned route is chosen within a
// bounded stretch of a shortest path while penalizing cumulative vertex,
// directed-edge, and reverse-edge load from routes assigned earlier in the
// same run. This is the simulator-native form of tools/generate_route_plans.py
// planner=tfo_gp and changes only the global route provider.
class TrafficFlowPlanner final : public Planner {
public:
    TrafficFlowPlanner(const GridMap& map, double max_stretch = 1.5,
                       double vertex_weight = 0.25, double edge_weight = 0.50,
                       double contraflow_weight = 2.0);
    [[nodiscard]] std::vector<CellId> plan(CellId start, CellId goal) override;
    void note_assigned_route(std::span<const CellId> route) override;

private:
    [[nodiscard]] const std::vector<std::int32_t>& distance_field(CellId goal);
    [[nodiscard]] double directed_load(CellId source, CellId destination) const;
    [[nodiscard]] std::size_t directed_index(CellId source, CellId destination) const;

    const GridMap& map_;
    double max_stretch_{1.5};
    double vertex_weight_{0.25};
    double edge_weight_{0.50};
    double contraflow_weight_{2.0};
    std::vector<double> vertex_load_;
    std::vector<double> edge_load_;
    std::unordered_map<CellId, std::vector<std::int32_t>> fields_;
    std::deque<CellId> field_order_;
};

std::unique_ptr<Planner> make_planner(PlannerKind kind, const GridMap& map, std::mt19937_64& rng);
std::unique_ptr<Planner> make_static_guidance_planner(
    const GridMap& map, std::uint64_t seed, double penalty = 0.20);

std::unique_ptr<Planner> make_traffic_flow_planner(
    const GridMap& map, double max_stretch = 1.5, double vertex_weight = 0.25,
    double edge_weight = 0.50, double contraflow_weight = 2.0);
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
