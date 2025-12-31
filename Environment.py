import os
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Sequence, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.AMR import AMR
from utils.Intersection import Intersection
from utils import Funct
from utils.task_generator import RandomGenerator, ScenGenerator
from utils.Controller import BFSPlanner, CBSPlanner
from utils.schedule_cache import CacheWriter


def _actions_to_paths_job(iid, inter, cache_db_path):
    """
    Function to be executed in a sub-process.
    
    - iid: Intersection ID (string)
    - inter: Intersection object (copy used only in this process)
    Returns:
      (iid, short_paths, target_exits)
    """
    inter.cache_db_path = cache_db_path
    short_paths, target_exits, cache_wb, cache_hit = inter.actions_to_paths()
    return iid, short_paths, target_exits, cache_wb, cache_hit


class ENV():
    def __init__(self, map_path, density, num_amrs, max_steps, planner, workers, cache_db_path, task_mode, scen_path, seed):
        super().__init__()
        """Initialize environment"""
        self.goal = set()

        self.seed = seed

        if self.seed != 0:
            self.py_rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
        else:
            self.py_rng = random.Random()
            self.np_rng = np.random.default_rng()

        self.scheduler_pool = ProcessPoolExecutor(max_workers=workers)

        self.time = 0
        
        self.map = self._load_map(map_path)
        bbox = self._get_goal_bbox(margin=0)  # Adjust margin if necessary
        walkable_tiles = self._count_walkable_in_bbox(bbox)
        if num_amrs > 0:
            self.num_amrs = num_amrs
            density = ( self.num_amrs / walkable_tiles ) * 100
        else:
            self.num_amrs = int((walkable_tiles * density) / 100)
        # print(f"\nMap width: {self.map.shape[1]}, Map height: {self.map.shape[0]}")
        # print(f"Walkable tiles (value 0): {walkable_tiles}")
        # print(f"Number of AMRs to spawn: {self.num_amrs}")
        # print(f"Density: {density:.2f}%")
        processed_intersections = self._find_intersections_and_build_graph()
        
        self.time = 0
        self.amr_list = {}
        self.max_steps = max_steps

        self.intersections: Dict[str, Intersection] = {}
        for iid, inter_info in processed_intersections.items():
            self.intersections[iid] = Intersection(
                inter_info['data'],
                inter_info['present_dirs'],
            )

        self.center_xs = sorted({I.center_x for I in self.intersections.values()})
        self.center_ys = sorted({I.center_y for I in self.intersections.values()})

        if planner == "bfs":
            self.planner = BFSPlanner(self.map, self.center_xs, self.center_ys, self.py_rng)
        elif planner == "cbs":
            self.planner = CBSPlanner(self.map, seed=self.seed, time_limit=60.0, center_xs=self.center_xs, center_ys=self.center_ys,)

        # Neighbor mapping between intersections
        self.iid_neighbors: dict[str, dict[str, str]] = {
            iid: dict(inter_info.get("neighbors", {}))
            for iid, inter_info in processed_intersections.items()
        }

        # Current number of AMRs per intersection (updated every step)
        self.iid_inside_counts: dict[str, int] = defaultdict(int)

        # Store scheduling capacity per intersection
        self.iid_scheduling_capacity: dict[str, int] = {}

        for iid, I in self.intersections.items():
            arm_lens = [len(I.lane_coords[d]) for d in I.dirs]
            cap_i = sum(arm_lens) - max(arm_lens)            
            I.scheduling_capacity = cap_i
            self.iid_scheduling_capacity[iid] = cap_i
            I.scheduling_capacity = cap_i
            I.available_count = cap_i

        # Mapping which intersection each cell belongs to
        self.cell2iids: Dict[tuple[int, int], list[str]] = defaultdict(list)

        # Event cells (intersection center + lane ends)
        self.event_center_cells = set()
        self.event_tip_cells = set()
        self.event_cells = set()
        self.event_cells2iid = {}

        # Build mapping
        for iid, I in self.intersections.items():
            center = (I.center_x, I.center_y)

            self.cell2iids[center].append(iid)
            self.event_cells.add(center)
            self.event_center_cells.add(center)
            self.event_cells2iid[center] = iid

            for d in I.dirs:
                coords = I.lane_coords[d]
                for cell in coords:
                    self.cell2iids[cell].append(iid)

                end_cell = coords[-1]
                self.event_cells.add(end_cell)
                self.event_tip_cells.add(end_cell)
                self.event_cells2iid[end_cell] = iid

        # Intersections in deadlock state
        self.deadlock_queue = []
        self.iid2sched: dict[str, set[int]] = defaultdict(set)
        self.deadlock_waiting_iids = set()

        # TaskGenerator
        if task_mode == "random":
            self.task_generator = RandomGenerator(self.map, num_tasks=self.num_amrs, goal_positions=self.goal, seed=self.seed)
        else:
            self.task_generator = ScenGenerator(scen_path=scen_path, num_tasks=self.num_amrs)

        # Color mapping
        self.color_map = Funct.Color_dict(6).dic

        self.use_scheduler = True

        self.completed_amr_steps = []

        self.completed_path_integrities: list[float] = []

        self.time_ms = []

        self.cache_db_path = cache_db_path
        self.cache_writer = CacheWriter(self.cache_db_path)  # DB file/table created here (Important!)
        self.cache_lookups = 0
        self.cache_hits = 0


    def reset(self):       
        if self.seed != 0:
            self.py_rng = random.Random(self.seed)
            self.np_rng = np.random.default_rng(self.seed)
        else:
            self.py_rng = random.Random()
            self.np_rng = np.random.default_rng()

        self.time = 0
        self.amr_list.clear()
        
        self.task_generator.start_new_episode()

        # Initialize internal state of all intersections
        for I in self.intersections.values():
            I.reset()

        self._spawn_amrs_from_task_gen()

        self.iid_inside_counts.clear()
        self.deadlock_queue = []
        self.iid2sched.clear()
        self.deadlock_waiting_iids.clear()

        self.completed_amr_steps.clear()
        self.completed_path_integrities.clear()
        self.time_ms.clear()

        self.cache_lookups = 0
        self.cache_hits = 0

        return

    
    def step(self):
        self.time += 1

        if self.task_generator.is_episode_done() or self.time > self.max_steps:
            return False
        
        if self.amr_list and all(amr.no_move_steps >= 10 for amr in self.amr_list.values()):
            return False

        # 1. Scheduler logic (deadlock detection and resolution)
        if self.use_scheduler:
            # (1) Check members per intersection
            check_iids = set()
            iid2members: dict[str, list[int]] = defaultdict(list)
            stalled_iids = set()
            self.iid_inside_counts.clear()

            for amr_id, amr_obj in self.amr_list.items():
                pos = tuple(amr_obj.pos)
                
                # Check which intersection area the current position belongs to
                if pos in self.cell2iids:
                    for iid in self.cell2iids[pos]:
                        iid2members[iid].append(amr_id)
                        self.iid_inside_counts[iid] += 1
                        
                # Check if deadlock check is needed (center + tip)
                if pos in self.event_cells:
                    iid = self.event_cells2iid[pos]
                    check_iids.add(iid)
            
            # In the first step, initialize free space counts for current and neighboring intersections
            if self.time == 1:
                # 1) Set available_count + create snapshot
                resv = {}
                for iid, I in self.intersections.items():
                    cap_i = self.iid_scheduling_capacity[iid]
                    count = self.iid_inside_counts.get(iid, 0)
                    I.available_count = cap_i - count
                    resv[iid] = cap_i - count

                # 2) Update neighbor available snapshot
                INF = 1e9
                for iid, I in self.intersections.items():
                    neigh_map = self.iid_neighbors.get(iid, {})
                    I.neighbor_available_count = {}

                    for d in I.dirs:
                        nid = neigh_map.get(d, None)
                        if nid is None:
                            I.neighbor_available_count[d] = INF
                        else:
                            I.neighbor_available_count[d] = resv[nid]

            # Check if intersection is in a stalled state
            for iid, members in iid2members.items():
                if not members:
                    continue
                if all(self.amr_list[aid].no_move_steps >= 1 for aid in members):
                    stalled_iids.add(iid)

            # Always include deadlock_waiting_iids in the check targets
            check_iids |= self.deadlock_waiting_iids | stalled_iids

            # (2-1) Unlock check (Attempt to resolve deadlock)
            for iid in list(self.deadlock_queue):
                scheduled_members = self.iid2sched[iid]
                still_active = False

                for mid in list(scheduled_members):
                    amr = self.amr_list.get(mid, None)
                    if amr is None:
                        scheduled_members.discard(mid)
                        continue
                    if amr.scheduling > 0:
                        still_active = True
                    else:
                        scheduled_members.discard(mid)
                
                if not still_active:
                    self.deadlock_queue.remove(iid)

            # Set of intersection IDs currently being scheduled
            active_iids = set(self.deadlock_queue)

            # (3-A) Deadlock check and collect "schedule candidate iid"
            iids_to_schedule: list[str] = []
            candidate_deadlocks: list[str] = []

            for iid in list(check_iids - active_iids):                
                I = self.intersections[iid]
                I.reset()

                # Register all AMRs in the intersection area
                for amr_id in iid2members.get(iid, []):
                    amr_obj = self.amr_list[amr_id]
                    I.register_amr(amr_obj)                

                # 3-1) Determine if deadlock exists
                is_deadlock = I.check_deadlock()
                if not is_deadlock:
                    # If no longer in deadlock, remove from pending candidates
                    self.deadlock_waiting_iids.discard(iid)
                    continue
                
                # 3-2) If neighbor intersection is being scheduled, skip this step and register in pending
                if self.has_active_neighbor(iid):
                    self.deadlock_waiting_iids.add(iid)
                    continue

                # If we reached here:
                #  - This iid is currently in a deadlock state
                #  - No active neighbor intersections -> scheduling can start in this step
                self.deadlock_waiting_iids.discard(iid)
                candidate_deadlocks.append(iid)
                
            # Sort candidate deadlock intersections by "ascending order of AMR count within the intersection"
            candidate_deadlocks.sort(
                key=lambda x: self.iid_inside_counts.get(x, 0),
                reverse=False,
            )

            # (3-A-2) Second pass: Determine actions_to_paths schedule in sorted order
            for iid in candidate_deadlocks:
                # At the time of the second pass, scheduling/distribution may have progressed,
                # so the deadlock_queue might have changed,
                # so check the neighbor active state once more.
                if self.has_active_neighbor(iid):
                    self.deadlock_waiting_iids.add(iid)
                    continue

                I = self.intersections[iid]
                cap_i = self.iid_scheduling_capacity[iid]

                # Continue if capacity is exceeded
                if self.iid_inside_counts.get(iid, 0) > cap_i:
                    I = self.intersections[iid]
                    self.deadlock_waiting_iids.add(iid)
                    continue

                if not self._allocate_neighbor_capacity(iid):
                    self.deadlock_waiting_iids.add(iid)
                    continue

                # If we reached here:
                #  - This iid is currently in a deadlock state
                #  - No active neighbor intersections -> scheduling can start in this step
                self.deadlock_waiting_iids.discard(iid)

                if iid not in active_iids:
                    self.deadlock_queue.append(iid)
                    active_iids.add(iid)

                # Execute actions_to_paths for this intersection
                iids_to_schedule.append(iid)

            # (3-B) Parallel execution of I.actions_to_paths() for iids_to_schedule
            futures = {}
            for iid in iids_to_schedule:
                I = self.intersections[iid]
                # Submit job to process pool
                fut = self.scheduler_pool.submit(_actions_to_paths_job, iid, I, self.cache_db_path)
                futures[fut] = iid

            # 1) Collect all results first (do not reflect absolute paths or put in cache here)
            buf = []
            for fut in as_completed(futures):
                iid = futures[fut]
                iid_ret, short_paths, target_exits, cache_wb, cache_hit = fut.result()
                buf.append((iid, iid_ret, short_paths, target_exits, cache_wb, cache_hit))

            # 2) Fix application order (based on iid)
            buf.sort(key=lambda x: x[0])  # x[0] == iid

            # 3) From now on, perform "result reflection + cache commit" in fixed order
            for iid, iid_ret, short_paths, target_exits, cache_wb, cache_hit in buf:
                self.cache_lookups += 1
                if cache_hit:
                    self.cache_hits += 1

                # if self.cache_lookups % 100 == 0:
                #     rate = 100.0 * self.cache_hits / self.cache_lookups
                #     print(f"[CACHE] lookups={self.cache_lookups} hits={self.cache_hits} hit_rate={rate:.2f}%")

                if cache_wb is not None:
                    key, blob = cache_wb
                    self.cache_writer.put_blob(key, blob)   

                # Insert scheduled path into AMR path
                for amr_id, short_path in short_paths.items():
                    if amr_id in self.amr_list:
                        amr_obj = self.amr_list[amr_id]
                        target_exit = target_exits[amr_id]
                        self.insert_scheduled_path(amr_obj, short_path, target_exit, iid)
                        self.iid2sched[iid].add(amr_id)

            for iid in stalled_iids:
                scheduling = False

                if iid in active_iids:
                    continue

                d, B, cycle = self.pick_edge_cycle_for_stalled(iid, stalled_iids, active_iids)
                if cycle is None:
                    continue
                
                I = self.intersections[iid]
                lane = I.lane_coords[d]

                lane_set = set(lane)

                edge_amrs = []
                for aid in iid2members.get(iid, []):
                    amr = self.amr_list[aid]
                    if amr.scheduling > 0:
                        scheduling = True
                        break
                    if tuple(amr.pos) in lane_set:
                        edge_amrs.append(amr)
                    if amr.pos == (I.center_x, I.center_y):
                        edge_amrs.append(amr)

                if scheduling:
                    continue

                if not edge_amrs:
                    continue

                for amr in edge_amrs:
                    self.build_and_insert_cycle_path(amr, iid, *cycle)   

        # 2. AMR Movement
        # (A) Initialize current position occupancy map
        current_occ = {amr.pos: amr.id for amr in self.amr_list.values()}

        # (B) Group separation
        normal_amrs = [a for a in self.amr_list.values() if a.scheduling == 0]
        scheduled_amrs = [a for a in self.amr_list.values() if a.scheduling > 0]

        # -------------------------------------------------------
        # [Phase 1] Normal AMRs move first
        # -------------------------------------------------------
        for amr in normal_amrs:
            cur_pos = amr.pos
            next_pos = amr.next_pos

            # ★ If entering a high-priority intersection tip or an intersection exceeding scheduling capacity -> wait in this step
            if self.block_intersection(cur_pos, next_pos, normal_only=True):
                amr.no_move_steps += 1
                continue

            # Existing collision/occupancy check
            if next_pos not in current_occ:
                if cur_pos in current_occ and current_occ[cur_pos] == amr.id:
                    del current_occ[cur_pos]

                self._update_available_on_move_success(cur_pos, next_pos)

                amr.move()
                current_occ[amr.pos] = amr.id
            else:
                amr.no_move_steps += 1

        # -------------------------------------------------------
        # [Phase 2] Scheduling blocking check
        # After normal robots have settled, check if scheduled robots can move
        # -------------------------------------------------------
        # Current set of normal robot positions
        normal_occ_pos = {amr.pos for amr in normal_amrs}

        # Identify blocked intersection IDs
        blocked_iids = set()

        for iid, members in self.iid2sched.items():
            # ✅ Iterate using a snapshot (list), modify the original set (members)
            for mid in list(members):
                amr = self.amr_list.get(mid, None)
                if amr is None:
                    members.discard(mid)
                    continue

                cur_pos = amr.pos
                next_pos = amr.next_pos

                # Block if entering the end of a high-priority intersection
                if self.block_intersection(cur_pos, next_pos, normal_only=False):
                    blocked_iids.add(iid)
                    break

                # Block if the next position is occupied by a normal robot
                if next_pos in normal_occ_pos:
                    blocked_iids.add(iid)
                    break
        
        # -------------------------------------------------------
        # [Phase 3] Scheduled robot movement
        # -------------------------------------------------------
        for amr in scheduled_amrs:
            # Check if the schedule I belong to is blocked
            is_blocked = False
            # Backtrack: Check which iid I belong to (iterate iid2sched)
            for iid, members in self.iid2sched.items():
                if amr.id in members:
                    if iid in blocked_iids:
                        is_blocked = True
                    break
            
            if is_blocked:
                # Skip because the entire group must wait
                amr.no_move_steps += 1
                continue

            # Perform movement
            if amr.pos in current_occ and current_occ[amr.pos] == amr.id:
                del current_occ[amr.pos]
            amr.move()
            current_occ[amr.pos] = amr.id


        # 3. Check completion and return info
        self._check_amr_completion()

        return self.make_info()
    

    def _allocate_neighbor_capacity(self, iid: str) -> bool:
        I = self.intersections[iid]
        dirs = list(I.dirs)

        neigh_map = self.iid_neighbors.get(iid, {})
        INF = 10**9

        # Neighbor intersection free space
        neigh_available = {}
        for d in dirs:
            nid = neigh_map.get(d)
            if nid is None:
                neigh_available[d] = INF
            else:
                neigh_available[d] = max(0, int(self.intersections[nid].available_count))

        inside_count = int(self.iid_inside_counts.get(iid, 0))

        initial_need = Counter()
        exit_need = Counter()
        for info in I.amr_intent_map.values():
            cur = info.get("current_arm")
            nxt = info.get("exit_arm")
            if cur in dirs:
                initial_need[cur] += 1
            if nxt in dirs:
                exit_need[nxt] += 1

        # ★ Capacity per direction: same as intersection arm length
        cap_list = [len(I.lane_coords[d]) for d in dirs]

        # Ensure quota does not exceed capacity (physical limit)
        per_stack_quota = []
        for i, d in enumerate(dirs):
            cap_d = cap_list[i]
            q_total = int(neigh_available[d]) + int(initial_need.get(d, 0))
            per_stack_quota.append(min(q_total, cap_d))

        I.stack_quota = per_stack_quota

        if sum(per_stack_quota) < inside_count:
            return False

        final_need = self.predict_final_stack_lengths(
            exit_need=exit_need,
            stack_capacities=cap_list,          # ★ here!
            per_stack_quota=per_stack_quota,
            order=dirs,
        )
        if final_need is None:
            return False

        # Apply delta
        for d in dirs:
            delta = int(final_need[d]) - int(initial_need.get(d, 0))
            nid = neigh_map.get(d)
            if nid is None:
                I.neighbor_available_count[d] = INF
                continue
            J = self.intersections[nid]
            J.available_count = int(J.available_count) - delta
            I.neighbor_available_count[d] = int(J.available_count)

        return True
    

    def predict_final_stack_lengths(
        self,
        exit_need: Counter,                             # {'N':7,'E':2,...}
        stack_capacities: Union[int, Sequence[int]] = 5,# ★ int or [cap0,cap1,...]
        per_stack_quota: Optional[Sequence[int]] = None,# ★ [q0,q1,...] (if none, use cap)
        order: Union[str, Sequence[str]] = "NESW",      # ★ "SEW" or ['S','E','W']
    ):
        # 0) Organize order
        order_list = list(order) if isinstance(order, str) else list(order)
        n = len(order_list)
        if n == 0:
            return {}

        # 1) Organize cap (per stack)
        if isinstance(stack_capacities, int):
            caps = [int(stack_capacities)] * n
        else:
            caps = [int(x) for x in stack_capacities]
            if len(caps) != n:
                return None

        # 2) Organize need / quota
        need = [int(exit_need.get(d, 0)) for d in order_list]

        if per_stack_quota is None:
            quota = caps[:]  # Default is cap
        else:
            q = [int(x) for x in per_stack_quota]
            if len(q) != n:
                return None
            quota = [max(0, min(q[i], caps[i])) for i in range(n)]  # Ensure quota does not exceed cap

        # 3) overflow type: need[i] > caps[i]
        overflow_types = {i for i in range(n) if need[i] > caps[i]}

        # 4) Assume default length in solved state
        lens = [0] * n
        for i in range(n):
            lens[i] = caps[i] if i in overflow_types else need[i]

        # 5) [Step 1] Distribute overflow excess (model of placing on top of non-overflow stacks)
        for t in range(n):  # order sequence is tie-break
            if t not in overflow_types:
                continue

            extra = need[t] - caps[t]
            for _ in range(extra):
                cands = [j for j in range(n)
                        if j != t and j not in overflow_types and lens[j] < caps[j]]
                if not cands:
                    return None

                min_len = min(lens[j] for j in cands)
                dst = next(j for j in range(n) if j in cands and lens[j] == min_len)  # tie: order
                lens[dst] += 1

        # 6) [Step 2] Adjust quota: if lens[i] > quota[i], move to another location
        #    (If unable to adjust, return None so allocate skips)
        while True:
            over = [(i, lens[i] - quota[i]) for i in range(n) if lens[i] > quota[i]]
            if not over:
                break

            max_over = max(k for _, k in over)
            src = next(i for i in range(n) if (lens[i] - quota[i]) == max_over)  # tie: order

            cands = [j for j in range(n)
                    if j != src and lens[j] < quota[j] and lens[j] < caps[j]]
            if not cands:
                return None

            def slack(j):
                return (quota[j] - lens[j], caps[j] - lens[j])

            best_sl = max(slack(j) for j in cands)
            dst = next(j for j in range(n) if j in cands and slack(j) == best_sl)  # tie: order

            lens[src] -= 1
            lens[dst] += 1

        return {order_list[i]: lens[i] for i in range(n)}


    
    def _find_4cycles_from_B(self, A, B):
        """
        Returns 4-cycles containing B.
        Cycle format: (B, C, D, E) => B-C-D-E-B
        """
        cycles = []
        neighbors_B = [x for x in self.iid_neighbors.get(B, {}).values() if x != A]

        # Select two neighbors of B (C, E) (i<j to avoid duplicates)
        for i in range(len(neighbors_B)):
            C = neighbors_B[i]
            for j in range(i+1, len(neighbors_B)):
                E = neighbors_B[j]

                # Common neighbors of C and E are candidates for D
                common = set(self.iid_neighbors.get(C, {}).values()) & set(self.iid_neighbors.get(E, {}).values())
                for D in common:
                    if D in (B, C, E):
                        continue
                    cycles.append((B, C, D, E))
        
        return cycles


    def pick_edge_cycle_for_stalled(self, A, stalled_iids, active_iids):
        """
        In stalled intersection A,
        - B is a non-stalled neighbor
        - A 4-cycle (B-C-D-E-B) containing B exists
        Randomly select one (dir_AB, B, (B,C,D,E)).

        Returns:
        (dir_AB, B, cycle_tuple) or (None, None, None)
        """
        I = self.intersections[A]

        candidates = []
        for d, B in self.iid_neighbors.get(A, {}).items():
            if B in stalled_iids or B in active_iids:
                continue

            # Only directions where A actually has an arm
            if d not in I.dirs:
                continue

            cycles = self._find_4cycles_from_B(A, B)
            if not cycles:
                continue

            candidates.append((d, B, cycles))

        if not candidates:
            return None, None, None

        d, B, cycles = self.py_rng.choice(candidates)
        cycle = self.py_rng.choice(cycles)

        # Randomly flip direction (clockwise/counter-clockwise)
        if self.py_rng.random() < 0.5:
            B, C, D, E = cycle
            cycle = (B, E, D, C)

        return d, B, cycle


    def build_and_insert_cycle_path(self, amr, iid, B, C, D, E):
        def center(iid):
            I = self.intersections[iid]
            return (I.center_x, I.center_y)

        Bc = center(B)
        start = tuple(amr.pos)
        waypoints = [Bc, center(C), center(D), center(E), center(B), start]

        full_path = [start]
        cur = start
        for wp in waypoints:
            seg = self.planner.plan_path(cur, wp)
            full_path.extend(seg[1:])  # Avoid duplicates
            cur = wp
        
        prefix = amr.path[:amr.path_cursor + 1]
        tail = amr.path[amr.path_cursor + 1:]

        if Bc in tail[:]:
            return  # Do not insert if already included in the path

        amr.path = prefix + full_path[1:] + tail

        if amr.path_cursor + 1 < len(amr.path):
            amr.next_pos = amr.path[amr.path_cursor + 1]
        else:
            amr.next_pos = amr.pos
        
        return True
        

    def insert_scheduled_path(self, amr, short_path, target_exit, iid):
        """
        Inserts the path (short_path) generated by the intersection scheduler into the current AMR path.

        - short_path : schedule path from amr.pos to merge_point inside the intersection
        - target_exit: the last cell (tip) of the exit lane the AMR 'originally' wants to go to

        Final path composition:
        prefix (path taken so far) +
        short_path[1:] (intersection internal schedule after current position) +
        bridge (merge_point -> rejoin point, patched with BFS) +
        continuation (tail after rejoin point in the original AMR path)
        """
        # Defensive code
        if not short_path or len(short_path) < 2:
            return

        merge_point = short_path[-1]

        # Last position in the original path followed so far (cursor position)
        if not (0 <= amr.path_cursor < len(amr.path)):
            return
        last_original_pos = amr.path[amr.path_cursor]

        # --- 1) bridge: BFS path from merge_point to rejoin_point ---
        #    Default is to use target_exit as the rejoin point,
        #    if target_exit is not found in the original path, handle by returning to last_original_pos
        bridge = [merge_point]
        rejoin_point = None       # Point to rejoin the original path at the end of the bridge
        continuation = []         # Original path tail after rejoin_point

        # (1) Attempt to find tail based on target_exit
        exit_idx = -1
        for i in range(amr.path_cursor + 1, len(amr.path)):
            if amr.path[i] == target_exit:
                exit_idx = i
                break

        if exit_idx != -1:
            # ✅ Normal case: target_exit exists in the original path
            rejoin_point = target_exit
            continuation = amr.path[exit_idx + 1:]
        else:
            # ❌ Case where target_exit is not found in the original path:
            #    Decided to return to last_original_pos and use the tail after it as is
            rejoin_point = last_original_pos
            continuation = amr.path[amr.path_cursor + 1:]

        # At this point, rejoin_point is
        #   - normal case: target_exit
        #   - exception case: last_original_pos
        # one of these, and continuation is the tail after rejoin_point

        # bridge calculation: merge_point -> rejoin_point
        if rejoin_point != merge_point:
            bridge = self.planner.plan_path(merge_point, rejoin_point)
        else:
            # If rejoin_point is the same as merge_point, bridge remains [merge_point]
            pass

        # --- 3) Construct new suffix ---
        new_suffix = []

        # (a) short_path: current position is in prefix, so start from [1:]
        new_suffix.extend(short_path[1:])

        # (b) bridge: start from [1:] to avoid merge_point duplication
        if len(bridge) > 1:
            new_suffix.extend(bridge[1:])

        # (c) original path tail after rejoin_point (or tail after last_original_pos)
        new_suffix.extend(continuation)

        # --- 4) Update AMR path/state ---
        prefix = amr.path[:amr.path_cursor + 1]
        amr.path = prefix + new_suffix

        # Scheduled segment length: short_path (remove one duplicate)
        sched_len = len(short_path) - 1
        amr.scheduling = sched_len

        # next_pos synchronization
        if len(amr.path) > amr.path_cursor + 1:
            amr.next_pos = amr.path[amr.path_cursor + 1]
        else:
            amr.next_pos = amr.pos

    
    def has_active_neighbor(self, iid):
        """
        Check if any of the neighbors of the intersection iid are currently in the deadlock_queue
        """
        for nid in self.iid_neighbors.get(iid, {}).values():
            if nid in self.deadlock_queue:
                return True
        return False


    def block_intersection(self, cur_pos, next_pos, normal_only=False) -> bool:
        """
        Returns True if movement from cur_pos to next_pos should be blocked by intersection-related policies (priority).

        Priority policy:
           - Current position must be 'outside' the intersection or at the intersection 'center'.
           - Next position must be at a 'lane end (tip)' of some intersection.
           - If the priority of the intersection being entered (= position in deadlock_queue) is higher than the current intersection -> True (blocked).

        Entry prohibited if intersection capacity is exceeded (when normal_only=True):
          - If the current number of internal AMRs in the intersection being entered exceeds scheduling capacity -> True (blocked).

        Priority:
          - Higher priority the earlier it is in the deadlock_queue (index 0, 1, 2, ...).
          - Intersections not in the deadlock_queue are treated as having the lowest priority.
        """
        # False if current position is inside the intersection but not at the center, or if next position is not at an intersection end
        is_cur_outside = cur_pos not in self.cell2iids
        if ((cur_pos not in self.event_center_cells and not is_cur_outside) 
            or next_pos not in self.event_tip_cells):
            return False

        # Intersections the current/next positions belong to (if at the center, current position belongs to exactly one intersection)
        cur_iid = self.cell2iids.get(cur_pos, [])
        cur_iid_set = set(self.cell2iids.get(cur_pos, []))
        next_iid_set = set(self.cell2iids.get(next_pos, []))

        entering_iid_set = next_iid_set - cur_iid_set
        entering_iid = next(iter(entering_iid_set))

        # normal_only mode: entry prohibited if intersection scheduling capacity is exceeded
        if normal_only:
            J = self.intersections[entering_iid]
            if J.available_count <= 0:
                return True

        # Priority policy   
        seq = self.deadlock_queue

        def priority(iid):
            try:
                return seq.index(iid)
            except ValueError:
                return len(seq)  # Lowest priority

        if is_cur_outside:
            cur_priority = len(seq)  # Lowest priority
        else:
            cur_priority = priority(cur_iid[0])

        # Find the highest priority among the intersections the next position belongs to
        next_priority = priority(entering_iid)

        if next_priority < cur_priority:
            return True

        return False
    

    def _update_available_on_move_success(self, cur_pos, next_pos):
        cur_set = set(self.cell2iids.get(tuple(cur_pos), []))
        nxt_set = set(self.cell2iids.get(tuple(next_pos), []))
        entering = nxt_set - cur_set   # Newly entering intersection
        leaving  = cur_set - nxt_set   # Leaving intersection

        if not entering and not leaving:
            return

        # leaving: space +1 (upper bound CAP)
        for iid in leaving:
            I = self.intersections[iid]
            I.available_count = min(I.available_count + 1, I.scheduling_capacity)

        # entering: space -1 (lower bound 0)
        for iid in entering:
            J = self.intersections[iid]
            J.available_count = max(0, J.available_count - 1)


    def _get_goal_bbox(self, margin: int = 0):
        """
        Set the x-range of goals as a bbox, while y is set to the full height.
        Returns: (x_min, y_min=0, x_max, y_max=H-1)
        """
        if not self.goal:
            return None

        xs = [x for x, _ in self.goal]
        x_min = min(xs) - margin
        x_max = max(xs) + margin

        H, W = self.map.shape
        x_min = max(0, x_min)
        x_max = min(W - 1, x_max)

        return x_min, 0, x_max, H - 1


    def _count_walkable_in_bbox(self, bbox) -> int:
        """
        LaCAM style:
        - Use only x-range
        - Exclude boundaries: x_min < x < x_max
        - Exclude goal tiles
        """
        H, W = self.map.shape

        # If no bbox, count walkable(0) in the entire map - exclude goals (only those that are 0)
        if bbox is None:
            walkable = int(np.count_nonzero(self.map == 0))
            goal_set = set(self.goal)
            goals_on_walkable = sum(
                1 for (x, y) in goal_set
                if 0 <= x < W and 0 <= y < H and self.map[y][x] == 0
            )
            return walkable - goals_on_walkable

        x_min, _, x_max, _ = bbox

        # Exclude boundaries: (x_min+1) ~ (x_max-1)
        left = x_min + 1
        right = x_max          # python slice end is exclusive, so it becomes x < x_max

        if left >= right:
            return 0

        # Count only the x-interval for all y
        area = self.map[:, left:right]
        walkable = int(np.count_nonzero(area == 0))

        # Exclude goals (only goals within the internal interval)
        goal_set = set(self.goal)
        goals_in_range = sum(
            1 for (x, y) in goal_set
            if left <= x < right and 0 <= y < H and self.map[y][x] == 0
        )

        return walkable - goals_in_range

    def _spawn_amrs_from_task_gen(self):
        """
        [Renamed and Task mode only]
        Receive new AMRs from RandomGenerator and add them to the environment.
        """
        gen = self.task_generator
        if not gen or not gen.should_spawn_next():
            return

        new_tasks = gen.get_next_task_pair(current_time=self.time)
        
        for task in new_tasks:
            amr_id = task['id']

            start_pos = tuple(task['start_pos'])
            goal_pos = tuple(task['goal_pos'])

            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _check_amr_completion(self):
        completed_amrs = []
        for amr_id, amr_obj in list(self.amr_list.items()):
            if amr_obj.pos == amr_obj.goal:
                completed_amrs.append(amr_id)

        for amr_id in completed_amrs:
            amr_obj = self.amr_list[amr_id]
            if amr_obj is not None:
                pi_pct = amr_obj.path_integrity_ratio()
                self.completed_path_integrities.append(pi_pct)
                self.completed_amr_steps.append(amr_obj.steps)
            self.task_generator.complete_task(amr_id)
            del self.amr_list[amr_id]


    def _spawn_amrs_from_stream_gen(self):
        """
        [Newly added function - Traffic mode only]
        Receive new AMRs from TrafficGenerator and add them to the environment.
        """
        gen = self.traffic_generator
        if not gen or not gen.should_spawn_next():
            return
        
        # TrafficGenerator12 does not have a current_time argument
        new_tasks = gen.get_next_task_pair()

        for task in new_tasks:
            amr_id = task['id']
            start_iid = task['intersection_id']
            start_dir = task['start_direction']
            goal_iid = task['goal_intersection_id']
            goal_dir = task['goal_direction']

            start_pos = self._direction_to_coords(start_dir, start_iid)
            goal_pos = self._direction_to_coords(goal_dir, goal_iid)

            if start_pos is None or goal_pos is None:
                continue
            
            # AMR creation and registration (corrected AMR constructor argument order)
            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _direction_to_coords(self, direction, intersection_ref):
        """
        direction: 'N'|'E'|'S'|'W'
        intersection_ref: intersection id string ("x{cx}y{cy}") or (cx,cy,lenN,lenE,lenS,lenW) tuple are both allowed
        """
        # 1) iid string -> Get spec from Intersection
        if isinstance(intersection_ref, str):
            I = self.intersections[intersection_ref]
            # If a dictionary like outer_entry_cells exists, use it first
            if hasattr(I, "outer_entry_cells") and direction in I.outer_entry_cells:
                return I.outer_entry_cells[direction]
            center_x, center_y = I.center_x, I.center_y
            len_N, len_E, len_S, len_W = I.len_N, I.len_E, I.len_S, I.len_W

        # 2) Backward compatibility: If spec comes as a tuple
        else:
            center_x, center_y, len_N, len_E, len_S, len_W = intersection_ref

        direction_map = {
            'N': (center_x, center_y - len_N - 1),
            'E': (center_x + len_E + 1, center_y),
            'S': (center_x, center_y + len_S + 1),
            'W': (center_x - len_W - 1, center_y),
        }
        return direction_map[direction]

    
    def _load_map(self, map_path):        
        if not os.path.isfile(map_path): raise FileNotFoundError(f"Map file not found: {map_path}")
        map_data = []
        with open(map_path, 'r') as f: lines = f.readlines()
        map_start = None
        for idx, line in enumerate(lines):
            if line.strip() == 'map': map_start = idx + 1; break
        if map_start is None: raise ValueError("Map data not found in file")
        for line in lines[map_start:]:
            row = []
            for c in line.strip():
                if c in ['@', 'T']: row.append(1)
                elif c in ['.', 'E', 'S']: 
                    row.append(0)
                    if c == "S":
                        self.goal.add((len(row)-1, len(map_data)))  # (x,y)
                else: raise ValueError(f"Invalid character in map file: {c}")
            if row: map_data.append(row)
        return np.array(map_data)

    def _find_intersection_center(self):
        # 3x3 patterns: 0=road, 1=wall
        plus4 = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        # T-shape (direction with one missing arm)
        t_noN = np.array([  # No top arm (only E/W/S open)
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])
        t_noE = np.array([  # No right arm (only N/W/S open)
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1]
        ])
        t_noS = np.array([  # No bottom arm (only N/E/W open)
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        t_noW = np.array([  # No left arm (only N/E/S open)
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 1]
        ])

        kernels = (plus4, t_noN, t_noE, t_noS, t_noW)

        # 3x3 sliding window
        windows = np.lib.stride_tricks.sliding_window_view(self.map, (3, 3))
        # Match each kernel and combine with OR
        match_any = np.zeros(windows.shape[:2], dtype=bool)
        for K in kernels:
            match_any |= np.all(windows == K, axis=(2, 3))

        # Window coordinates -> Center coordinates (sliding offset +1)
        centers = (np.argwhere(match_any) + 1).tolist()
        return centers
        
    def _ray_len(self, r, c, dr, dc):
        H, W = self.map.shape
        length = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W and self.map[rr][cc] == 0:
            if (cc, rr) in self.goal:
                break
            if dr != 0:
                left_wall  = (cc - 1 < 0) or (self.map[rr][cc - 1] == 1)
                right_wall = (cc + 1 >= W) or (self.map[rr][cc + 1] == 1)
                if not (left_wall and right_wall): break
            else:
                up_wall   = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall and down_wall): break

            length += 1
            rr += dr
            cc += dc
        return length
    
    def _find_intersections_and_build_graph(self):
        centers_rc = self._find_intersection_center()
        centers_xy = [(c, r) for r, c in centers_rc]

        center_xy_to_data = {}
        for c, r in centers_xy:
            len_N = self._ray_len(r, c, -1, 0)
            len_S = self._ray_len(r, c,  1, 0)
            len_E = self._ray_len(r, c,  0, 1)
            len_W = self._ray_len(r, c,  0,-1)

            # ★ Allow 4-way/3-way: Recognized as an intersection only if there are 3 or more arms
            present = {d for d, L in zip("NESW", [len_N, len_E, len_S, len_W]) if L > 0}
            if len(present) >= 3:
                center_xy_to_data[(c, r)] = (c, r, len_N, len_E, len_S, len_W, present)

        processed_intersections = {}
        for (c, r), tup in center_xy_to_data.items():
            c, r, len_N, len_E, len_S, len_W, present = tup
            current_iid = f'x{c}y{r}'

            # ★ Calculate neighbors only for existing arms
            neighbors_map = {}
            if 'N' in present:
                t = (c, r - len_N - 1)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['N'] = f'x{nc}y{nr}'
            if 'E' in present:
                t = (c + len_E + 1, r)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['E'] = f'x{nc}y{nr}'
            if 'S' in present:
                t = (c, r + len_S + 1)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['S'] = f'x{nc}y{nr}'
            if 'W' in present:
                t = (c - len_W - 1, r)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['W'] = f'x{nc}y{nr}'

            processed_intersections[current_iid] = {
                'data': (c, r, len_N, len_E, len_S, len_W),
                'neighbors': neighbors_map,
                # ↓ Pass for easy use in mask/state zero-padding in later steps
                'present_dirs': present,
            }
        return processed_intersections

    
    def is_arm_outgoing_clear(self, iid: str, d: str) -> bool:
        I = self.intersections[iid]

        # (Optional) 3-way response: Prohibit non-existent arms
        present = getattr(I, "present_dirs", set(I.lane_coords.keys()))
        if d not in present:
            return False

        # 1) Prohibit if there is an 'outgoing flow' on that arm
        has_outgoing = bool(getattr(I, "outgoing", {}).get(d, False))
        if has_outgoing:
            return False

        # 2) Prohibit if intersection is in deadlock ← deadlock_queue normalization
        dq = self.deadlock_queue or []
        if dq and isinstance(dq[0], tuple):
            dead_iids = {x for (x, _) in dq}
        else:
            dead_iids = set(dq)
        if iid in dead_iids:
            return False

        # (Recommended) 3) Check if the arm tip (outer entry) is a road and empty
        if hasattr(I, "outer_entry_cells") and d in I.outer_entry_cells:
            tip = I.outer_entry_cells[d]
        else:
            cx, cy = I.center_x, I.center_y
            if   d == "N": tip = (cx, cy - I.len_N - 1)
            elif d == "E": tip = (cx + I.len_E + 1, cy)
            elif d == "S": tip = (cx, cy + I.len_S + 1)
            else:          tip = (cx - I.len_W - 1, cy)

        H, W = self.map.shape
        tx, ty = tip
        if not (0 <= tx < W and 0 <= ty < H):
            return False
        if self.map[ty][tx] == 1:
            return False
        if any(a.pos == tip for a in self.amr_list.values()):
            return False

        return True

    
    def _update_and_check_stagnation(self) -> bool:
        """
        Detect stagnation/oscillation based on recent global position signatures.
        Returns True if early termination is required.
        """
        if self.time < self._stg_min_time:
            self._sig_hist.clear()
            return False
        if not self.amr_list:
            self._sig_hist.clear()
            return False

        # Global signature: sorted tuple of (amr_id, x, y) tuples
        sig = tuple(sorted((aid, amr.pos[0], amr.pos[1]) for aid, amr in self.amr_list.items()))
        self._sig_hist.append(sig)

        # 1) Idle: Recent N are all identical
        idle = False
        if len(self._sig_hist) >= self._stg_idle_win:
            lastN = list(self._sig_hist)[-self._stg_idle_win:]
            idle = all(s == lastN[0] for s in lastN)

        # 2) Oscillation: Recent M are in ABABAB form (two signatures alternating)
        osc = False
        if len(self._sig_hist) >= self._stg_osc_win:
            w = self._stg_osc_win
            lastM = list(self._sig_hist)[-w:]
            if lastM[0] != lastM[1]:
                osc = all(lastM[i] == lastM[i % 2] for i in range(w))

        if idle or osc:
            return True

        return False
    

    # --- [Adapter functions for GUI integration] ---
    def Get_AMR(self):
        """Function to allow GUI to get the AMR list"""
        return self.amr_list

    def get_active_tasks(self):
        """Function to allow GUI to get the AMR goal positions"""
        return {amr_id: amr_obj.goal for amr_id, amr_obj in self.amr_list.items()}

    def make_info(self):
        """
        [Modified] Calculates and returns all information needed for the GUI.
        Explicitly handles 'task' and 'traffic' modes.
        """
        # --- 2. Calculate statistical information according to mode ---
        progress = self.task_generator.get_progress()
        completed_tasks = progress.get('completed_total', 0)
        total_tasks = progress.get('spawned_total', 0)

        # Calculate Success Rate
        success_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # Calculate throughput (per minute)
        throughput = (completed_tasks / self.time * 60) if self.time > 0 else 0.0

        active_pi = []
        for amr_obj in self.amr_list.values():
            active_pi.append(amr_obj.path_integrity_ratio())
        all_pi = self.completed_path_integrities + active_pi
        avg_pi = float(np.mean(all_pi)) if all_pi else 0.0

        # --- 3. Collect detailed information of currently active AMRs ---
        active_amr_details = {}
        for amr_id, amr_obj in self.amr_list.items():
            active_amr_details[amr_id] = {
                "steps": amr_obj.steps,
            }

        # --- 4. Aggregate and return final information ---
        return {
            "success_rate": success_rate,
            "throughput": throughput,
            "active_amrs": active_amr_details,
            "avg_path_integrity": avg_pi,
            "time": self.time,
        }
