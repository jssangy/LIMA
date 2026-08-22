#pragma once
#include "lima/core/agent.hpp"
#include <cstddef>
#include <span>
#include <vector>
namespace lima {
// Local two-cell handshake for a boundary service terminal. Before an agent
// leaves its inward predecessor for G, it leases both cells. The predecessor
// is therefore an acknowledged return credit until the agent comes back.
// Routing and intersection solving remain independent of this lease table.
class TerminalEgressReservations {
public:
    void resize(std::size_t agent_count, std::size_t cell_count);
    void begin_cycle(std::span<const Agent> agents, bool enabled);
    [[nodiscard]] bool try_acquire(AgentId agent, CellId terminal, CellId inward);
    [[nodiscard]] bool allows(AgentId agent, CellId destination) const noexcept;
    [[nodiscard]] CellId return_cell(AgentId agent, CellId position) const noexcept;
    [[nodiscard]] CellId terminal_cell(AgentId agent) const noexcept;
    [[nodiscard]] CellId inward_cell(AgentId agent) const noexcept;
private:
    struct Lease { CellId terminal{kInvalidCell}; CellId inward{kInvalidCell}; };
    std::vector<Lease> leases_;
    std::vector<AgentId> terminal_owner_;
    std::vector<AgentId> inward_owner_;
    void release(AgentId agent) noexcept;
    void clear() noexcept;
};
}  // namespace lima
