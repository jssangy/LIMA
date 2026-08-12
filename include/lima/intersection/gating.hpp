#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"
#include "lima/planning/planner.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace lima {

// Which local-solvability capacity bound an intersection advertises.
// SumMinusMax is the shipped implementation; SumMinusMaxPlusOne matches the
// manuscript's Eq. (9) exactly and exists to reconcile the two (finding M1).
enum class CapacityFormula : std::uint8_t { SumMinusMax, SumMinusMaxPlusOne };

struct IsolationConfig {
    CapacityFormula formula{CapacityFormula::SumMinusMax};
    int cap{-1};     // operational ceiling on the bound; -1 disables (experiment E11)
    int margin{0};   // reserve slack subtracted from the bound (gating study)
};

// Discharge behavior variants for the gating study; defaults reproduce the
// shipped recirculation policy exactly.
struct DischargeConfig {
    bool all_arms{false};              // reroute members on every arm, not just the escape arm
    bool allow_stalled_neighbor{false};// permit loops that start at a stalled neighbor (jam erosion)
    bool deterministic_cycle{false};   // least-loaded cycle choice instead of random draws
};

[[nodiscard]] int scheduling_capacity(const Intersection& intersection, const IsolationConfig& config);

// Discharge gating: when every member of an intersection has stalled, selected
// agents are routed through a temporary recirculation loop over four
// neighboring intersections and rejoin their original route afterwards.
class RecirculationDischarge {
public:
    RecirculationDischarge() = default;
    explicit RecirculationDischarge(DischargeConfig config) : config_(config) {}
    struct Context {
        const IntersectionTopology& topology;
        std::span<Agent> agents;
        const std::vector<std::vector<AgentId>>& members;
        const std::vector<bool>& stalled;
        const std::vector<std::uint8_t>& deadlock_active;
        Planner& planner;
        std::mt19937_64& rng;
    };
    struct Event {
        IntersectionId intersection{-1};
        int rerouted{};
        std::size_t loop_cells{};
        std::vector<AgentId> agent_ids;
    };

    [[nodiscard]] std::vector<Event> run(const Context& context) const;

private:
    DischargeConfig config_{};
};

}  // namespace lima
