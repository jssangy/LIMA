#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"

#include <cstdint>
#include <array>
#include <span>
#include <vector>

namespace lima {

enum class AdmitLookaheadMode : std::uint8_t { Off, Hard, Threshold, Ratio, Differential };
enum class AimdSignalMode : std::uint8_t { Local, NeighborMax, NeighborMean, Trend };
enum class AdmitCreditMode : std::uint8_t { Off, Equal, Demand, DeficitRoundRobin };

struct AdmissionInformationConfig {
    AdmitLookaheadMode lookahead{AdmitLookaheadMode::Off};
    double lookahead_parameter{0.0};
    AimdSignalMode aimd_signal{AimdSignalMode::Local};
    double aimd_signal_parameter{0.0};
    AdmitCreditMode credit{AdmitCreditMode::Off};
    double credit_parameter{0.0};
};

struct AdmissionInformationEvent {
    const char* mechanism{};
    const char* event{};
    IntersectionId source{-1};
    IntersectionId target{-1};
    std::uint64_t count{};
    std::uint32_t bytes{};
    std::uint16_t hops{};
    std::uint16_t delay{};
};

struct AdmissionCreditRequest {
    AgentId agent{-1};
    IntersectionId intersection{-1};
    std::size_t direction{};
    std::uint32_t age{};
};

// Information-axis extension for the Admission Controller.  It owns only
// 1-hop controller state and deterministic grant bookkeeping; movement and
// hard-capacity enforcement remain in Simulator.
class AdmissionInformationAxis {
public:
    explicit AdmissionInformationAxis(AdmissionInformationConfig config = {}) : config_(config) {}

    void resize(std::size_t intersections, std::size_t agents);
    [[nodiscard]] bool lookahead_blocks(IntersectionId source, IntersectionId downstream,
                                        std::span<const std::size_t> stale_loads,
                                        std::span<const int> capacities,
                                        std::span<const int> availability);
    [[nodiscard]] bool aimd_congested(IntersectionId source, bool local_congested,
                                      const IntersectionTopology& topology,
                                      std::span<const std::size_t> stale_loads,
                                      std::span<const int> capacities);
    void prepare_credits(std::span<const AdmissionCreditRequest> requests,
                         std::span<const int> availability,
                         std::span<const int> reserve);
    [[nodiscard]] bool credit_allows(AgentId agent, IntersectionId entering) const;
    void note_credit_consumed(AgentId agent, IntersectionId entering);
    [[nodiscard]] std::vector<AdmissionInformationEvent> take_events();

    [[nodiscard]] const AdmissionInformationConfig& config() const noexcept { return config_; }

private:
    AdmissionInformationConfig config_{};
    std::vector<double> previous_neighbor_signal_;
    std::vector<std::uint32_t> increasing_steps_;
    std::vector<std::array<double, 4>> credit_deficit_;
    std::vector<IntersectionId> grants_;
    std::vector<AdmissionInformationEvent> events_;
};

}  // namespace lima
