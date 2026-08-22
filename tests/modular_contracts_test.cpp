#include "lima/intersection/admission.hpp"
#include "lima/intersection/admission_information.hpp"
#include "lima/intersection/gating.hpp"
#include "lima/planning/planner.hpp"
#include "lima/simulation/terminal_egress.hpp"

#include <cassert>
#include <cmath>
#include <string_view>
#include <vector>

namespace {

class StubPlanner final : public lima::Planner {
public:
    std::vector<lima::CellId> plan(
        const lima::CellId start, const lima::CellId goal) override {
        last_start = start;
        last_goal = goal;
        if (start == 9 && goal == 2) return {9, 8, 2};
        return {};
    }

    lima::CellId last_start{lima::kInvalidCell};
    lima::CellId last_goal{lima::kInvalidCell};
};

void test_suffix_repair() {
    StubPlanner planner;
    const std::vector<lima::CellId> reference{0, 1, 2, 3};
    const auto repair = lima::repair_to_reference_suffix(planner, 9, reference, 1);
    assert(repair.has_value());
    assert((repair->route == std::vector<lima::CellId>{9, 8, 2, 3}));
    assert(repair->rejoin == 2);
    assert(repair->reference_rejoin_index == 2);
    assert(repair->bridge_edges == 2);
    assert(planner.last_start == 9 && planner.last_goal == 2);
}

void test_operational_capacity() {
    lima::Intersection intersection;
    for (std::size_t arm = 0; arm < intersection.arms.size(); ++arm) {
        intersection.arms[arm].resize(5, static_cast<lima::CellId>(arm));
    }
    lima::IsolationConfig config;
    config.formula = lima::CapacityFormula::SumMinusMax;
    assert(lima::scheduling_capacity(intersection, config) == 15);

    intersection.arms[2].resize(4);
    intersection.arms[3].resize(4);
    assert(lima::scheduling_capacity(intersection, config) == 13);
}

void test_acknowledged_aimd() {
    lima::AcknowledgedAimdAdmission controller;
    const std::vector<int> capacities{4};
    controller.reset(3, capacities,
        {.multiplicative_decrease = 0.5, .additive_increase = 0.5});

    const std::vector<lima::AimdIntersectionObservation> free{{4, 1, false, false}};
    std::vector<lima::AimdAdmissionRequest> requests{{0, 0, 2}, {1, 0, 1}};
    controller.update(free, requests);
    assert(controller.granted(0, 0));
    assert(controller.granted(1, 0));

    // Multiplicative decrease limits only new grants. Existing acknowledged
    // work in flight remains valid across the congestion signal.
    const std::vector<lima::AimdIntersectionObservation> stalled{{4, 4, true, false}};
    requests.push_back({2, 0, 0});
    controller.update(stalled, requests);
    assert(std::abs(controller.windows()[0] - 2.0) < 1e-12);
    assert(controller.granted(0, 0));
    assert(controller.granted(1, 0));
    assert(!controller.granted(2, 0));

    controller.acknowledge_entry(0, 0);
    requests = {{1, 0, 3}, {2, 0, 2}};
    controller.update(free, requests);
    assert(std::abs(controller.windows()[0] - 2.5) < 1e-12);
    assert(controller.granted(1, 0));
    assert(controller.granted(2, 0));

    // An execution failure is a stutter step, not structural congestion.
    const std::vector<lima::AimdIntersectionObservation> delayed{{4, 2, true, true}};
    controller.update(delayed, requests);
    assert(std::abs(controller.windows()[0] - 2.5) < 1e-12);
}

void test_directional_credit_advertisement() {
    lima::AdmissionInformationAxis axis({
        .credit = lima::AdmitCreditMode::Equal,
    });
    axis.resize(1, 3);
    const std::vector<lima::AdmissionCreditRequest> requests{
        {0, 0, 0, 3}, {1, 0, 0, 2}, {2, 0, 1, 1},
    };
    const std::vector<int> availability{3};
    const std::vector<int> reserve{0};
    axis.prepare_credits(requests, availability, reserve);

    assert(axis.credit_allows(0, 0));
    assert(axis.credit_allows(1, 0));
    assert(axis.credit_allows(2, 0));

    std::uint64_t issued = 0;
    std::uint64_t advertised = 0;
    for (const auto& event : axis.take_events()) {
        if (std::string_view(event.event) == "issued") issued += event.count;
        if (std::string_view(event.event) == "advertised") advertised += event.count;
    }
    assert(issued == 3);
    // A3 sends one scalar per active direction, not one message per credit.
    assert(advertised == 2);
}

void test_terminal_egress_reservation() {
    lima::TerminalEgressReservations reservations;
    reservations.resize(3, 12);
    std::vector<lima::Agent> agents(3);
    for (std::size_t i = 0; i < agents.size(); ++i) {
        agents[i].id = static_cast<lima::AgentId>(i);
        agents[i].active = true;
    }
    agents[0].position = 5;
    agents[0].goal = 6;
    agents[0].route = {5, 6};
    reservations.begin_cycle(agents, true);
    assert(reservations.try_acquire(0, 6, 5));
    assert(reservations.allows(0, 5) && reservations.allows(0, 6));
    assert(!reservations.allows(1, 5) && !reservations.allows(1, 6));
    assert(reservations.virtual_occupancy_cell(0, 5) == lima::kInvalidCell);
    assert(reservations.preserves_capacity_credit(0, 5, 6));
    agents[0].position = 6;
    agents[0].goal = 2;
    agents[0].route = {6, 5, 4, 3, 2};
    reservations.begin_cycle(agents, true);
    assert(reservations.return_cell(0, 6) == 5);
    assert(reservations.virtual_occupancy_cell(0, 6) == 5);
    assert(reservations.preserves_capacity_credit(0, 6, 5));
    assert(!reservations.preserves_capacity_credit(0, 6, 4));
    assert(!reservations.try_acquire(1, 6, 5));
    agents[0].position = 5;
    agents[0].route = {5, 4, 3, 2};
    reservations.begin_cycle(agents, true);
    assert(reservations.return_cell(0, 5) == lima::kInvalidCell);
    assert(reservations.allows(1, 5) && reservations.allows(1, 6));
    assert(reservations.try_acquire(1, 6, 5));
    reservations.begin_cycle(agents, false);
    assert(reservations.allows(2, 5) && reservations.allows(2, 6));
}

}  // namespace

int main() {
    test_operational_capacity();
    test_suffix_repair();
    test_acknowledged_aimd();
    test_directional_credit_advertisement();
    test_terminal_egress_reservation();
    return 0;
}
