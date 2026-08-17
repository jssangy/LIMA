#pragma once

#include "lima/intersection/admission_information.hpp"
#include "lima/intersection/topology.hpp"

#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace lima {

enum class RecirculationProbeMode : std::uint8_t {
    Off,
    Detect,
    BreakAtSlack,
    BreakAtLongArm,
};

struct RecirculationProbeConfig {
    RecirculationProbeMode mode{RecirculationProbeMode::Off};
    std::uint16_t ttl{4};
    std::uint16_t activation_age{1};
};

// Chandy-Misra-Haas style edge chasing over the intersection wait-for graph.
// Every call advances a probe by exactly one adjacent-controller link.
class RecirculationProbeDetector {
public:
    explicit RecirculationProbeDetector(RecirculationProbeConfig config = {}) : config_(config) {}

    void resize(std::size_t intersections);
    void step(const IntersectionTopology& topology, const std::vector<bool>& stalled,
              std::span<const IntersectionId> wait_for);
    [[nodiscard]] const std::vector<std::vector<IntersectionId>>& cycles() const noexcept {
        return detected_cycles_;
    }
    [[nodiscard]] std::vector<AdmissionInformationEvent> take_events();
    [[nodiscard]] const RecirculationProbeConfig& config() const noexcept { return config_; }

private:
    struct Probe {
        IntersectionId origin{-1};
        IntersectionId current{-1};
        std::uint16_t remaining{};
        std::vector<IntersectionId> path;
    };

    RecirculationProbeConfig config_{};
    std::vector<std::uint32_t> stall_age_;
    std::vector<std::vector<IntersectionId>> detected_cycles_;
    std::vector<Probe> probes_;
    std::unordered_set<IntersectionId> active_origins_;
    std::vector<AdmissionInformationEvent> events_;
};

}  // namespace lima
