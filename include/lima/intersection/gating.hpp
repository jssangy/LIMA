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
// SumMinusMax is the frozen operational envelope. SumMinusMaxPlusOne is kept
// only as a sensitivity/legacy-manuscript audit variant.
enum class CapacityFormula : std::uint8_t { SumMinusMax, SumMinusMaxPlusOne };

// Local recirculation-loop selection policies.  Every dynamic policy consumes
// only source state, adjacent-module state, or the one-cycle-stale aggregates
// already carried by the discharge exchange.
enum class DischargePolicy : std::uint8_t {
    Legacy,
    Composite,
    Random,
    LeastLoaded,
    MaxSlack,
    Rotor,
    Shortest,
    PowerOfTwo,
    Backpressure,
    Balanced,
    Demand,
};

struct IsolationConfig {
    CapacityFormula formula{CapacityFormula::SumMinusMax};
    int cap{-1};       // operational ceiling on the bound; -1 disables (experiment E11)
    int margin{0};     // reserve slack subtracted from the bound (gating study)
    int hysteresis{0}; // admit only while available > hysteresis (0 = baseline)
};

// Discharge behavior variants for the gating study; defaults reproduce the
// shipped recirculation policy exactly.
struct DischargeConfig {
    bool all_arms{false};              // reroute members on every arm, not just the escape arm
    bool allow_stalled_neighbor{false};// permit loops that start at a stalled neighbor (jam erosion)
    // Defaults since 2026-08-13 (tournament winner "dcombo"): escape toward the
    // 1-hop neighbor with the most admission slack, and within that direction
    // prefer the loop over the least-loaded intersections (one-cycle-stale
    // relayed counts).  Fully deterministic; -19.6% makespan on the hard cells.
    bool deterministic_cycle{true};
    bool avail_weighted{true};
    double partial_stall{1.0};         // waiting-member fraction that marks an intersection stalled
    DischargePolicy policy{DischargePolicy::Legacy};
    double weight{1.0};                // neighbor backlog weight for the balanced policy
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
        const std::vector<int>* available{nullptr};  // per-intersection admission slack (1-hop knowledge)
        // Previous-cycle occupancy counts: models load aggregates relayed over
        // the existing adjacent-module exchange (one cycle stale), keeping
        // load-aware cycle choice within the limited-communication rule.
        const std::vector<std::size_t>* stale_loads{nullptr};
    };
    struct Event {
        IntersectionId intersection{-1};
        int rerouted{};
        std::size_t loop_cells{};
        std::vector<AgentId> agent_ids;
        std::vector<std::size_t> agent_loop_cells;
        std::vector<std::uint8_t> agent_loop_closed;
    };

    [[nodiscard]] std::vector<Event> run(const Context& context) const;

private:
    DischargeConfig config_{};
    mutable std::vector<std::size_t> rotor_cursor_;
};

}  // namespace lima
