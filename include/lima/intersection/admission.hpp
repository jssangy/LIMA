#pragma once

#include "lima/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lima {

// Admission policies are local Intersection-Agent strategies. Their control
// state is independent of the hard physical-capacity accounting performed at
// movement commit time.
enum class AdmissionPolicy : std::uint8_t {
    Static,
    FractionalReserve,
    RequestProportional,
    Backpressure,
    NeighborPressure,
    Aimd,
    Red,
    Blue,
    Rem,
    Avq,
    Codel,
    Pi,
    Pie,
    TokenBucket,
    Sotl,
    Choke,
    QueueCsma,
    StochasticFairBlue,
    FqCodel,
    LongestQueue,
    OldestRequest,
    RoundRobin,
    DeficitRoundRobin,
};

struct AdmissionConfig {
    AdmissionPolicy policy{AdmissionPolicy::Static};
    // Policy-specific knobs. Their interpretation is recorded by the CLI
    // manifest; zero selects the policy default.
    double parameter{0.0};
    double secondary{0.0};
    double tertiary{0.0};
};

struct AimdAdmissionRequest {
    AgentId agent{kNoAgent};
    IntersectionId intersection{-1};
    std::uint32_t wait_steps{};
};

struct AimdIntersectionObservation {
    int capacity{};
    int occupied{};
    bool stalled{};
    bool execution_delayed{};
};

struct AcknowledgedAimdConfig {
    double multiplicative_decrease{0.5};
    double additive_increase{0.25};
};

// Acknowledged AIMD controls entry work in flight. A grant is an eligibility
// token, not a cell reservation: the ordinary capacity gate still authorizes
// the physical entry. Keeping these notions separate makes the capacity
// invariant independent of controller tuning and execution delays.
class AcknowledgedAimdAdmission {
public:
    void reset(std::size_t agent_count, std::span<const int> capacities,
               AcknowledgedAimdConfig config);

    // Reconciles persistent grants with this step's request set, updates each
    // local window from structural congestion/acknowledgements, and grants
    // oldest requests first. Existing grants are never revoked by AIMD.
    void update(std::span<const AimdIntersectionObservation> intersections,
                std::span<const AimdAdmissionRequest> requests);

    [[nodiscard]] bool granted(AgentId agent, IntersectionId intersection) const noexcept;
    void acknowledge_entry(AgentId agent, IntersectionId intersection) noexcept;

    [[nodiscard]] std::span<const double> windows() const noexcept { return windows_; }
    [[nodiscard]] std::span<const IntersectionId> grant_targets() const noexcept {
        return grant_target_;
    }

private:
    double beta_{0.5};
    double alpha_{0.25};
    std::vector<int> capacities_;
    std::vector<double> windows_;
    std::vector<IntersectionId> grant_target_;
    std::vector<std::size_t> outstanding_;
    std::vector<std::size_t> acknowledged_;
    std::vector<std::vector<AimdAdmissionRequest>> requests_by_intersection_;
};

}  // namespace lima
