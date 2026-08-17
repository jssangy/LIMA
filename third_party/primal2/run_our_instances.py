#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_our_instances.py -- inference harness for pretrained PRIMAL2 (one-shot)
on our MovingAI-style MAPF instances.

Maps   : MovingAI .map, chars . S E G traversable, @ T blocked.
Scens  : MovingAI .scen (bucket, map, w, h, start_x(col), start_y(row),
         goal_x(col), goal_y(row), opt_len); we take the first N tasks.
Policy : pretrained one-shot checkpoint (model_primal2_oneshot/model-97500.cptk),
         greedy decoding, one ACNet forward per agent per step, per-agent LSTM state.
Sem.   : DISAPPEAR-AT-TARGET. When an agent reaches its goal it is removed from
         the world (PRIMAL2's own isOneShot branch in MAPFEnv.step_all removes it
         from state/goals_map; we additionally remove it from collision checking
         and from other agents' observations -- see README_ours.md).

Everything runs TF1 session-style on CPU with fixed seeds.

Usage (from the PRIMAL2 repo root, conda env `primal2`):
    python run_our_instances.py --map warehouse_10_20 --scen warehouse-10-20_s0 -n 26
See README_ours.md for details and deviations.
"""
from __future__ import division, print_function

import argparse
import copy
import os
import random
import sys
import time
import types
from collections import deque

# ------------------------------------------------------------------ CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- stub out gym's pyglet/OpenGL rendering module BEFORE importing Env_Builder.
# Env_Builder.py:12 does `from gym.envs.classic_control import rendering`, which
# needs a display. We never render, so an empty module is enough (Python's
# from-import falls back to sys.modules for the submodule lookup).
_rendering_stub = types.ModuleType("gym.envs.classic_control.rendering")
sys.modules["gym.envs.classic_control.rendering"] = _rendering_stub

import numpy as np
import tensorflow as tf

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

import Env_Builder
from Env_Builder import World
from Primal2Env import Primal2Env
from Primal2Observer import Primal2Observer
from Ray_ACNet import ACNet

TRAVERSABLE = frozenset(".SEG")
BLOCKED = frozenset("@T")

DEFAULT_MAP_DIR = os.path.expanduser("~/lima-dev/data/maps")
DEFAULT_SCEN_DIR = os.path.expanduser("~/lima-dev/data/scenarios")
DEFAULT_MODEL_DIR = os.path.join(REPO_DIR, "model_primal2_oneshot")

MASK64 = (1 << 64) - 1
TRACE_SCALE = 1 << 53


def splitmix64(value):
    """Unsigned SplitMix64, byte-for-byte compatible with LIMA's delay trace."""
    value = (value + 0x9e3779b97f4a7c15) & MASK64
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK64
    return (value ^ (value >> 31)) & MASK64


def command_delayed(instance_seed, agent_zero_based, timestep, probability):
    """Counter-based common trace keyed only by seed, agent, and timestep."""
    if probability <= 0.0:
        return False
    counter = (int(instance_seed) ^ 0xa0761d6478bd642f) & MASK64
    counter ^= ((int(agent_zero_based) + 1) * 0xd2b74407b1ce6e93) & MASK64
    counter ^= ((int(timestep) + 1) * 0xca5a826395121157) & MASK64
    sample = splitmix64(counter) >> 11
    threshold = int(float(probability) * TRACE_SCALE)
    return sample < threshold


# =========================================================================
# instance loading (MovingAI)
# =========================================================================

def load_map(path):
    """Parse a MovingAI .map into PRIMAL2's world format: int array, -1 obstacle,
    0 free. The grid is padded with a 1-cell obstacle ring (all coordinates shift
    by +1): PRIMAL2's own map generator always walls the border, and its corridor
    / collision code crashes on traversable cells that touch the array edge
    (cross_3030 has 120 such cells)."""
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0].split()[0] == "type", "not a MovingAI map: %s" % path
    h = int(lines[1].split()[1])
    w = int(lines[2].split()[1])
    assert lines[3].strip() == "map"
    rows = lines[4:4 + h]
    assert len(rows) == h and all(len(r) >= w for r in rows), "map body size mismatch"
    grid = -np.ones((h + 2, w + 2), dtype=int)
    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch in TRAVERSABLE:
                grid[r + 1, c + 1] = 0
            elif ch not in BLOCKED:
                raise ValueError("unknown map char %r at (%d,%d)" % (ch, r, c))
    return grid, h, w


def load_scen(path, n):
    """Return the first n tasks as [(start_rc, goal_rc), ...] in PADDED grid
    coordinates (row = y + 1, col = x + 1)."""
    tasks = []
    with open(path) as f:
        body = f.read().splitlines()
    assert body[0].lower().startswith("version"), "not a MovingAI scen: %s" % path
    for line in body[1:]:
        if not line.strip():
            continue
        p = line.split("\t")
        sx, sy, gx, gy = int(p[4]), int(p[5]), int(p[6]), int(p[7])
        tasks.append(((sy + 1, sx + 1), (gy + 1, gx + 1)))
        if len(tasks) == n:
            break
    if len(tasks) < n:
        raise SystemExit("BLOCKER: scen %s has only %d tasks, need %d" % (path, len(tasks), n))
    return tasks


def validate_instance(grid, tasks):
    """Fail fast, precisely, if the instance cannot be expressed. Duplicate GOALS
    are allowed (handled by per-agent goal bookkeeping); duplicate STARTS or
    start==goal are not expressible / not meaningful in PRIMAL2's world."""
    starts = [t[0] for t in tasks]
    goals = [t[1] for t in tasks]
    for i, (s, g) in enumerate(tasks):
        for name, cell in (("start", s), ("goal", g)):
            if grid[cell] != 0:
                raise SystemExit("BLOCKER: task %d %s %s is not traversable" % (i, name, cell))
        if s == g:
            raise SystemExit("BLOCKER: task %d has start == goal %s" % (i, s))
    if len(set(starts)) != len(starts):
        raise SystemExit("BLOCKER: duplicate start cells within the first N tasks "
                         "(PRIMAL2's state map holds one agent per cell)")
    # connectivity: every start/goal must sit in one traversable component
    seen = np.zeros(grid.shape, dtype=bool)
    dq = deque([starts[0]])
    seen[starts[0]] = True
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if grid[nx, ny] == 0 and not seen[nx, ny]:
                seen[nx, ny] = True
                dq.append((nx, ny))
    for i, (s, g) in enumerate(tasks):
        if not seen[s] or not seen[g]:
            raise SystemExit("BLOCKER: task %d start %s / goal %s not connected to task 0" % (i, s, g))


# =========================================================================
# exact fast replacement for Env_Builder.getAstarDistanceMap
# =========================================================================

def fast_bfs_distance_map(world_map, start, goal, isDiagonal=False):
    """Drop-in replacement for Env_Builder.getAstarDistanceMap (performance only,
    output-identical). The original runs A* from `goal` to exhaustion with a
    consistent heuristic on a unit-cost 4-connected grid, so every cell reachable
    from the goal receives its exact shortest-path distance; BFS computes the same
    values. Like the original, cells not reached keep their input map value and
    `start` plays no role."""
    assert not isDiagonal
    dist_map = np.array(world_map).copy()
    h, w = dist_map.shape
    gx, gy = int(goal[0]), int(goal[1])
    seen = np.zeros((h, w), dtype=bool)
    seen[gx, gy] = True
    dq = deque([(gx, gy, 0)])
    while dq:
        x, y, d = dq.popleft()
        dist_map[x, y] = d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and dist_map[nx, ny] != -1:
                seen[nx, ny] = True
                dq.append((nx, ny, d + 1))
    return dist_map


# =========================================================================
# one-shot world / env / observer with true disappear-at-target semantics
# =========================================================================

class OneShotWorld(World):
    """World with (a) deterministic manual agent/goal initialization that allows
    duplicate goal cells, and (b) collision checking that fully ignores agents
    that already reached their goal."""

    def __init__(self, map_generator, num_agents, isDiagonal, starts, goals):
        self._manual_starts = starts   # {agentID: (r, c)}
        self._manual_goals = goals     # {agentID: (r, c)}
        super(OneShotWorld, self).__init__(map_generator, num_agents=num_agents,
                                           isDiagonal=isDiagonal)

    def init_agents_and_goals(self):
        """Replaces World.init_agents_and_goals/_put_agents/put_goals for manual
        one-shot instances. Differences vs stock code, both required:
        1. duplicate goal cells allowed: goals_map is a single-channel array (one
           agentID per cell, last writer wins) and is kept only as a vestigial
           representation; each agent's authoritative goal is agent.goal_pos, and
           the own-goal observation channel is fixed per-agent in OneShotObserver.
        2. no random `next_goal`: stock put_goals samples a random future goal for
           every agent (a training-time artifact). Under one-shot disappear
           semantics there is no journey after the goal, so next_goal := goal_pos
           and next_distanceMap := distanceMap, i.e. the observation's projected
           A* path shows the agent staying at its goal. Also makes init
           independent of RNG."""
        for agentID in range(1, self.num_agents + 1):
            agent = self.agents[agentID]
            start = self._manual_starts[agentID]
            goal = self._manual_goals[agentID]
            agent.ID = agentID
            assert self.state[start] == agentID, \
                "state map / start mismatch for agent %d" % agentID
            agent.move(start)
            agent.goal_pos = goal
            self.goals_map[goal[0], goal[1]] = agentID
            agent.distanceMap = Env_Builder.getAstarDistanceMap(self.state, start, goal)
            agent.next_goal = goal
            agent.next_distanceMap = agent.distanceMap

    def get_corridors(self):
        """Copy of World.get_corridors (Env_Builder.py:289) with one bug fix in the
        endpoint-adjacency lookups of the sorting pass. Stock code (lines 343-348
        and 361-366) picks the first neighbor of an endpoint whose corridor id
        matches, then looks it up in the corridor's interior 'Positions' list. When
        a 2-cell corridor's two exit junctions are adjacent to each other (e.g.
        warehouse_10_20 rack-gap corridor [(3,137),(4,137)] with endpoints (3,138)
        and (4,138)), the OTHER ENDPOINT (type 2) matches first and .index() raises
        ValueError. Fix: additionally require corridor_map[position][1] == 1
        (interior corridor cell). Identical behavior wherever stock code worked."""
        corridor_count = 1
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i, j] >= 0:
                    self.corridor_map[(i, j)] = [-1, 0]
                else:
                    self.corridor_map[(i, j)] = [-1, -1]
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                positions = self.blank_env_valid_neighbor(i, j)
                if (positions.count(None)) == 2 and (i, j) not in self.visited:
                    allowed = self.check_for_singular_state(positions)
                    if not allowed:
                        continue
                    self.corridors[corridor_count] = {}
                    self.corridors[corridor_count]['Positions'] = [(i, j)]
                    self.corridor_map[(i, j)] = [corridor_count, 1]
                    self.corridors[corridor_count]['EndPoints'] = []
                    self.visited.append((i, j))
                    for num in range(4):
                        if positions[num] is not None:
                            self.visit(positions[num][0], positions[num][1], corridor_count)
                    corridor_count += 1
        for k in range(1, corridor_count):
            if k in self.corridors:
                if len(self.corridors[k]['EndPoints']) == 2:
                    self.corridors[k]['DeltaX'] = {}
                    self.corridors[k]['DeltaY'] = {}
                    pos_a = self.corridors[k]['EndPoints'][0]
                    pos_b = self.corridors[k]['EndPoints'][1]
                    self.corridors[k]['DeltaX'][pos_a] = (pos_a[0] - pos_b[0])
                    self.corridors[k]['DeltaX'][pos_b] = -1 * self.corridors[k]['DeltaX'][pos_a]
                    self.corridors[k]['DeltaY'][pos_a] = (pos_a[1] - pos_b[1])
                    self.corridors[k]['DeltaY'][pos_b] = -1 * self.corridors[k]['DeltaY'][pos_a]
            else:
                print('Weird2')

        for t in range(1, corridor_count):
            positions = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][0][0],
                                                      self.corridors[t]['EndPoints'][0][1])
            for position in positions:
                if position is not None and self.corridor_map[position][0] == t \
                        and self.corridor_map[position][1] == 1:  # <-- fix: interior cells only
                    break
            index = self.corridors[t]['Positions'].index(position)

            if index == 0:
                pass
            if index != len(self.corridors[t]['Positions']) - 1:
                temp_list = self.corridors[t]['Positions'][0:index + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)

            elif index == len(self.corridors[t]['Positions']) - 1 and len(self.corridors[t]['EndPoints']) == 2:
                positions2 = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][1][0],
                                                           self.corridors[t]['EndPoints'][1][1])
                for position2 in positions2:
                    if position2 is not None and self.corridor_map[position2][0] == t \
                            and self.corridor_map[position2][1] == 1:  # <-- fix: interior cells only
                        break
                index2 = self.corridors[t]['Positions'].index(position2)
                temp_list = self.corridors[t]['Positions'][0:index2 + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index2 + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)
                self.corridors[t]['Positions'].reverse()
            else:
                if len(self.corridors[t]['EndPoints']) == 2:
                    print("Weird3")

            self.corridors[t]['StoppingPoints'] = []
            if len(self.corridors[t]['EndPoints']) == 2:
                position_first = self.corridors[t]['Positions'][0]
                position_last = self.corridors[t]['Positions'][-1]
                self.corridors[t]['StoppingPoints'].append([position_first[0], position_first[1]])
                self.corridors[t]['StoppingPoints'].append([position_last[0], position_last[1]])
            else:
                position_first = self.corridors[t]['Positions'][0]
                self.corridors[t]['StoppingPoints'].append([position[0], position[1]])
                self.corridors[t]['StoppingPoints'].append(None)
        return

    def CheckCollideStatus(self, movement_dict):
        """Copy of World.CheckCollideStatus (Env_Builder.py:673) restricted to
        LIVE agents. Stock code iterates over all agents, so a finished ("done")
        one-shot agent kept occupying its goal cell inside the cell-wise collision
        check (blocking any higher-ID agent from entering it) even after step_all
        removed it from the map. Skipping done agents completes the
        disappear-at-target semantics. Return dicts are keyed by live agents only;
        step_all only reads entries of non-done agents, so that is sufficient."""
        if self.isDiagonal is True:
            raise NotImplemented
        live = [aid for aid in range(1, self.num_agents + 1) if self.agents[aid].dones == 0]
        Assumed_newPos_dict = {}
        newPos_dict = {}
        status_dict = {agentID: None for agentID in live}
        not_checked_list = list(live)

        # detect env collision
        for agentID in live:
            direction_vector = Env_Builder.action2dir(movement_dict[agentID])
            newPos = Env_Builder.tuple_plus(self.getPos(agentID), direction_vector)
            Assumed_newPos_dict.update({agentID: newPos})
            if newPos[0] < 0 or newPos[0] > self.state.shape[0] or newPos[1] < 0 \
                    or newPos[1] > self.state.shape[1] or self.state[newPos] == -1:
                status_dict[agentID] = -1
                newPos_dict.update({agentID: self.getPos(agentID)})
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)

        # detect swap collision
        for agentID in copy.deepcopy(not_checked_list):
            collided_ID = self.state[Assumed_newPos_dict[agentID]]
            if collided_ID != 0:  # someone is standing on the assumed pos
                if Assumed_newPos_dict[collided_ID] == self.getPos(agentID):  # swap
                    if status_dict[agentID] is None:
                        status_dict[agentID] = -2
                        newPos_dict.update({agentID: self.getPos(agentID)})
                        Assumed_newPos_dict[agentID] = self.getPos(agentID)
                        not_checked_list.remove(agentID)
                    if status_dict[collided_ID] is None:
                        status_dict[collided_ID] = -2
                        newPos_dict.update({collided_ID: self.getPos(collided_ID)})
                        Assumed_newPos_dict[collided_ID] = self.getPos(collided_ID)
                        not_checked_list.remove(collided_ID)

        # detect cell-wise collision
        for agentID in copy.deepcopy(not_checked_list):
            other_agents_dict = copy.deepcopy(Assumed_newPos_dict)
            other_agents_dict.pop(agentID)
            if Assumed_newPos_dict[agentID] in newPos_dict.values():
                status_dict[agentID] = -3
                newPos_dict.update({agentID: self.getPos(agentID)})
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
            elif Assumed_newPos_dict[agentID] in other_agents_dict.values():
                other_coming_agents = Env_Builder.get_key(Assumed_newPos_dict,
                                                          Assumed_newPos_dict[agentID])
                other_coming_agents.remove(agentID)
                if agentID < min(other_coming_agents):
                    status_dict[agentID] = 1 \
                        if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
                    newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
                    not_checked_list.remove(agentID)
                else:
                    status_dict[agentID] = -3
                    newPos_dict.update({agentID: self.getPos(agentID)})
                    Assumed_newPos_dict[agentID] = self.getPos(agentID)
                    not_checked_list.remove(agentID)

        # the rest are valid actions
        for agentID in copy.deepcopy(not_checked_list):
            status_dict[agentID] = 1 \
                if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
            newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
            not_checked_list.remove(agentID)
        assert not not_checked_list

        return status_dict, newPos_dict


class OneShotObserver(Primal2Observer):
    """Primal2Observer with two one-shot fixes:
    1. per-agent own-goal channel: goals_map holds one agentID per cell, so with
       duplicate goal cells most agents would never see their own goal in the
       goal_map channel. We temporarily stamp the querying agent's ID on its goal
       cell while its observation is built (world.goals_map is used ONLY for the
       own-goal channel inside _get; the other-goals channel uses per-agent
       getGoal), then restore.
    2. get_astar_map skips agents that already reached their goal: their rows stay
       all-zero, so vanished agents contribute no phantom future path to their
       neighbors' observations. This also avoids an IndexError in the stock path
       expansion for an agent standing on its goal with next_goal == goal."""

    def _get(self, agent_id, all_astar_maps):
        g = self.world.agents[agent_id].goal_pos
        prev = self.world.goals_map[g[0], g[1]]
        if self.world.agents[agent_id].dones == 0:
            self.world.goals_map[g[0], g[1]] = agent_id
        try:
            return super(OneShotObserver, self)._get(agent_id, all_astar_maps)
        finally:
            self.world.goals_map[g[0], g[1]] = prev

    def get_astar_map(self):
        """Copy of Primal2Observer.get_astar_map (Primal2Observer.py) with a
        single change: done agents (dones > 0) are skipped, leaving zero maps."""

        def get_single_astar_path(distance_map, start_position, path_len):

            def get_astar_one_step(position):
                next_astar_cell = []
                h = self.world.state.shape[0]
                w = self.world.state.shape[1]
                for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    new_pos = Env_Builder.tuple_plus(position, direction)
                    if 0 < new_pos[0] <= h and 0 < new_pos[1] <= w:
                        if distance_map[new_pos] == distance_map[position] - 1 \
                                and distance_map[new_pos] >= 0:
                            next_astar_cell.append(new_pos)
                return next_astar_cell

            path_counter = 0
            astar_list = [[start_position]]
            while path_counter < path_len:
                last_step_cells = astar_list[-1]
                next_step_cells = []
                for cells_per_step in last_step_cells:
                    new_cell_list = get_astar_one_step(cells_per_step)
                    if not new_cell_list:
                        astar_list.pop(0)
                        return astar_list
                    next_step_cells.extend(new_cell_list)
                next_step_cells = list(set(next_step_cells))
                astar_list.append(next_step_cells)
                path_counter += 1

            astar_list.pop(0)
            return astar_list

        astar_maps = {}
        for agentID in range(1, self.world.num_agents + 1):
            astar_maps.update(
                {agentID: np.zeros([self.num_future_steps,
                                    self.world.state.shape[0], self.world.state.shape[1]])})

            if self.world.agents[agentID].dones > 0:  # <-- one-shot: vanished agent
                continue

            distance_map0 = self.world.agents[agentID].distanceMap
            start_pos0 = self.world.agents[agentID].position
            astar_path = get_single_astar_path(distance_map0, start_pos0, self.num_future_steps)

            if not len(astar_path) == self.num_future_steps:
                distance_map1 = self.world.agents[agentID].next_distanceMap
                start_pos1 = self.world.agents[agentID].goal_pos
                astar_path.extend(
                    get_single_astar_path(distance_map1, start_pos1,
                                          self.num_future_steps - len(astar_path)))

            for i in range(self.num_future_steps - len(astar_path)):
                astar_path.extend([[astar_path[-1][-1]]])

            assert len(astar_path) == self.num_future_steps
            for step in range(self.num_future_steps):
                for cell in astar_path[step]:
                    astar_maps[agentID][step, cell[0], cell[1]] = 1

        return np.asarray([astar_maps[i] for i in range(1, self.world.num_agents + 1)])


class OneShotPrimal2Env(Primal2Env):
    """Primal2Env wired to OneShotWorld. Everything else (listValidActions with
    PRIMAL2's corridor conventions, step_all's isOneShot removal-at-goal branch)
    is inherited unchanged."""

    def __init__(self, observer, map_generator, starts, goals, num_agents):
        self._starts = starts
        self._goals = goals
        super(OneShotPrimal2Env, self).__init__(observer=observer,
                                                map_generator=map_generator,
                                                num_agents=num_agents,
                                                IsDiagonal=False,
                                                isOneShot=True)

    def set_world(self):
        self.world = OneShotWorld(self.map_generator, num_agents=self.num_agents,
                                  isDiagonal=self.IsDiagonal,
                                  starts=self._starts, goals=self._goals)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)


# =========================================================================
# main
# =========================================================================

def resolve_paths(args):
    map_path = args.map
    if os.sep not in map_path and "/" not in map_path:
        map_path = os.path.join(DEFAULT_MAP_DIR, map_path + ("" if map_path.endswith(".map") else ".map"))
    scen_path = args.scen
    if os.sep not in scen_path and "/" not in scen_path:
        base = scen_path if scen_path.endswith(".scen") else scen_path + ".scen"
        scen_dir = base.rsplit("_s", 1)[0]
        scen_path = os.path.join(DEFAULT_SCEN_DIR, scen_dir, base)
    return map_path, scen_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True,
                    help="map name (resolved under ~/lima-dev/data/maps) or path")
    ap.add_argument("--scen", required=True,
                    help="scen name, e.g. warehouse-10-20_s0 (resolved under "
                         "~/lima-dev/data/scenarios/<dir>/) or path")
    ap.add_argument("-n", "--num-agents", type=int, required=True,
                    help="number of agents = first N scen tasks")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="step cap; default 4 * map perimeter = 4 * 2*(H+W) of the "
                         "unpadded map")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--delay-prob", type=float, default=0.0,
                    help="counter-hash-v1 move-command delay probability")
    ap.add_argument("--delay-seed", type=int, default=0,
                    help="instance seed for the common (agent,timestep) delay trace")
    ap.add_argument("--model", default=DEFAULT_MODEL_DIR,
                    help="checkpoint directory (default: model_primal2_oneshot)")
    ap.add_argument("--obs-size", type=int, default=11,
                    help="FOV size the checkpoint was trained with")
    ap.add_argument("--future-steps", type=int, default=3,
                    help="num A* future-step channels the checkpoint was trained with")
    ap.add_argument("--no-fast-astar", action="store_true",
                    help="use the stock O(V^2) pure-python A* distance maps instead "
                         "of the output-identical BFS replacement")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="print a progress line every K steps (0 = never)")
    args = ap.parse_args()
    if args.delay_prob < 0.0 or args.delay_prob > 1.0:
        ap.error("--delay-prob must be in [0,1]")

    # determinism ---------------------------------------------------------
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    sys.setrecursionlimit(100000)  # corridor detection is recursive

    if not args.no_fast_astar:
        Env_Builder.getAstarDistanceMap = fast_bfs_distance_map

    map_path, scen_path = resolve_paths(args)
    t0 = time.time()
    grid, H, W = load_map(map_path)
    tasks = load_scen(scen_path, args.num_agents)
    validate_instance(grid, tasks)

    n = args.num_agents
    max_steps = args.max_steps if args.max_steps is not None else 4 * 2 * (H + W)

    starts = {i + 1: tasks[i][0] for i in range(n)}
    goals = {i + 1: tasks[i][1] for i in range(n)}
    state0 = grid.copy()
    for aid, s in starts.items():
        state0[s] = aid

    def map_generator():
        return state0.copy(), None  # goals_map filled by OneShotWorld

    observer = OneShotObserver(observation_size=args.obs_size,
                               num_future_steps=args.future_steps)
    env = OneShotPrimal2Env(observer=observer, map_generator=map_generator,
                            starts=starts, goals=goals, num_agents=n)
    t_env = time.time() - t0
    print("[setup] env built in %.1fs (map %dx%d padded to %dx%d, %d agents, "
          "%d corridors, max_steps=%d)"
          % (t_env, H, W, grid.shape[0], grid.shape[1], env.num_agents,
             len(env.world.corridors), max_steps))

    # network -------------------------------------------------------------
    t0 = time.time()
    tf.reset_default_graph()
    num_channels = 8 + args.future_steps
    net = ACNet("global", 5, None, False, num_channels, args.obs_size, "global",
                GLOBAL_NETWORK=False)
    config = tf.ConfigProto(allow_soft_placement=True, device_count={"GPU": 0})
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.model)
    assert ckpt is not None, "no checkpoint found in %s" % args.model
    ckpt_path = os.path.join(args.model, os.path.basename(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt_path)
    print("[setup] checkpoint %s restored in %.1fs" % (ckpt_path, time.time() - t0))

    # episode -------------------------------------------------------------
    # As in Worker.run_episode_multithreaded: one forward pass per agent per step,
    # per-agent LSTM state, PRIMAL2's listValidActions masking; but GREEDY
    # (argmax over valid actions) instead of sampling.
    obs = env._observe()
    rnn_states = {i: net.state_init for i in range(1, n + 1)}
    live = set(range(1, n + 1))
    steps = 0
    command_delays = 0
    t_ep = time.time()
    while live and steps < max_steps:
        movement = {}
        for aid in sorted(live):
            s = obs[aid]
            a_dist, rnn = sess.run(
                [net.policy, net.state_out],
                feed_dict={net.inputs: [s[0]],
                           net.goal_pos: [s[1]],
                           net.state_in[0]: rnn_states[aid][0],
                           net.state_in[1]: rnn_states[aid][1]})
            rnn_states[aid] = rnn
            valid = env.listValidActions(aid, s)
            movement[aid] = valid[int(np.argmax(a_dist.flatten()[valid]))]
        if args.delay_prob > 0.0:
            timestep = steps + 1
            for aid in sorted(live):
                if movement[aid] != 0 and command_delayed(
                        args.delay_seed, aid - 1, timestep, args.delay_prob):
                    movement[aid] = 0
                    command_delays += 1
        obs, _ = env.step_all(movement)
        steps += 1
        for aid in list(live):
            if env.world.agents[aid].dones > 0:
                live.discard(aid)
        if args.progress_every and steps % args.progress_every == 0:
            print("[t=%4d] completed %d/%d, %.1fs elapsed"
                  % (steps, n - len(live), n, time.time() - t_ep))
            sys.stdout.flush()
    wall = time.time() - t_ep

    completed = n - len(live)
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    scen_name = os.path.splitext(os.path.basename(scen_path))[0]
    print("SUMMARY map=%s scen=%s N=%d solved_fraction=%.4f completed=%d/%d "
          "steps=%d wall_s=%.1f failures=%d failure_prob=%g delay_trace=counter-hash-v1"
          % (map_name, scen_name, n, completed / float(n), completed, n, steps, wall,
             command_delays, args.delay_prob))


if __name__ == "__main__":
    main()
