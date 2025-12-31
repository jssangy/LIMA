from itertools import chain

from utils.sch import schedule


class Intersection:
    def __init__(self, intersection_data, present_dirs):
        self.center_x, self.center_y, self.len_N, self.len_E, self.len_S, self.len_W = intersection_data
        self.id = f'x{self.center_x}y{self.center_y}'

        if present_dirs is None:
            present_dirs = {d for d,L in zip("NESW",[self.len_N,self.len_E,self.len_S,self.len_W]) if L>0}
        self.present_dirs = set(present_dirs)
        self.dirs = [d for d in "NESW" if d in self.present_dirs]

        self.lane_coords = {}
        if 'N' in self.present_dirs:
            self.lane_coords['N'] = [(self.center_x, self.center_y - i) for i in range(1, self.len_N + 1)]
        if 'E' in self.present_dirs:
            self.lane_coords['E'] = [(self.center_x + i, self.center_y) for i in range(1, self.len_E + 1)]
        if 'S' in self.present_dirs:
            self.lane_coords['S'] = [(self.center_x, self.center_y + i) for i in range(1, self.len_S + 1)]
        if 'W' in self.present_dirs:
            self.lane_coords['W'] = [(self.center_x - i, self.center_y) for i in range(1, self.len_W + 1)]

        self.all_lane_coords = set(chain.from_iterable(self.lane_coords.values()))
        self.all_lane_coords.add((self.center_x, self.center_y))

        # Event-based AGV object tracking
        self.amr_intent_map = {}            # {amr_id: {'amr_obj': amr, 'current_arm': 'N', 'exit_arm': 'S'}}
        self.is_deadlock = False
        self.paths = {}                 # {amr_id: [(x,y), ...]}
        self.target_exits = {}          # {amr_id: (x,y)}  # Original exit tip coordinates for each AMR

        self.scheduling_capacity = 0         # Maximum number of AMRs that can be scheduled
        self.available_count = 0             # Number of available spaces for AMRs in the intersection
        self.neighbor_available_count = {}   # Number of available spaces per adjacent intersection
        self.stack_quota = []                # Stack quota per direction


    def reset(self):
        self.amr_intent_map = {}
        self.is_deadlock = False
        self.paths = {}
        self.target_exits = {}

    
    def register_amr(self, amr):
        path = amr.path
        if not path:
            raise ValueError("AMR path is empty; cannot register intent.")

        center = (self.center_x, self.center_y)
        current_arm_direction = None
        exit_arm_direction = None
        exit_cell = None

        # Search in forward direction
        for i in range(amr.path_cursor, len(path) - 1):
            if path[i] == center:
                next_pos_index = i + 1
                exit_cell = path[next_pos_index]
                break

        for direction, coords in self.lane_coords.items():
            if amr.pos == center:
                current_arm_direction = "C"
            if amr.pos in coords:
                current_arm_direction = direction
            if exit_cell is not None and exit_cell in coords:
                exit_arm_direction = direction

        if exit_cell is None:
            exit_arm_direction = current_arm_direction

        if current_arm_direction is None or exit_arm_direction is None:
            raise ValueError("Could not determine current or exit arm direction for AMR.")

        self.amr_intent_map[amr.id] = {
            'amr_obj': amr,
            'current_arm': current_arm_direction,
            'exit_arm': exit_arm_direction
        }

        # --- Additional: Set target_exits immediately at register_amr ---
        tip_cell = None
        if exit_arm_direction in self.lane_coords:
            coords = self.lane_coords[exit_arm_direction]
            if coords:
                tip_cell = coords[-1]  # The very end tip of the arm
        self.target_exits[amr.id] = tip_cell

    def check_deadlock(self):
        if self.check_cycle_deadlock():
            return True
        if self.check_center_deadlock():
            return True
        if self.check_swap_deadlock():
            return True
        return False


    def check_cycle_deadlock(self):
        dirs = self.dirs
        adj = {d: set() for d in dirs}

        pos2rec = {}
        for rec in self.amr_intent_map.values():
            a = rec.get('amr_obj')
            if a: pos2rec[a.pos] = rec

            cur = rec['current_arm']
            nxt = rec['exit_arm']
            if cur in adj and nxt in adj and cur != nxt:
                adj[cur].add(nxt)

        visited = set()
        onstack = set()

        def dfs(u: str) -> bool:
            visited.add(u)
            onstack.add(u)
            for v in adj[u]:
                if v not in visited:
                    if dfs(v):
                        return True
                elif v in onstack:
                    return True
            onstack.remove(u)
            return False

        for u in adj.keys():
            if u not in visited and dfs(u):
                self.is_deadlock = True
                return True
        
        inline_conflict = False
        for d in dirs:
            coords = self.lane_coords.get(d, [])
            if not coords:
                continue
            # Scan adjacent pairs
            for i in range(len(coords) - 1):
                front_pos = coords[i]       # Cell closer to center (rank = i+1)
                behind_pos = coords[i + 1]  # Cell immediately behind it (rank = i+2)

                rec_front = pos2rec.get(front_pos)
                rec_behind = pos2rec.get(behind_pos)
                if not rec_front or not rec_behind:
                    continue

                # Both must be in the same lane d
                if rec_front.get('current_arm') != d or rec_behind.get('current_arm') != d:
                    continue

                nxt_front  = rec_front.get('exit_arm')
                nxt_behind = rec_behind.get('exit_arm')

                # If front cell intends 'outward' and behind cell intends 'inward (center)', they conflict
                if nxt_front == d and nxt_behind != d:
                    inline_conflict = True
                    break
            if inline_conflict:
                break

        self.is_deadlock = inline_conflict
        return self.is_deadlock

    
    def check_center_deadlock(self):
        """
        Detect deadlock based on the AMR at the center.

        1) If there is no AMR with current_arm == 'C' -> Not a deadlock
        2) If there is, its exit_arm = exit_dir
        3) Iterate through amr_intent_map again:
        - If an AMR with current_arm == exit_dir exists, and
        - That AMR's exit_arm != current_arm
            -> Deadlock because it intends to conflict with the center AMR
        """
        # 1. Find AMR at the center
        center_exit = None
        for aid, rec in self.amr_intent_map.items():
            if rec.get('current_arm') == 'C':
                center_exit = rec.get('exit_arm')
                break

        # No center AMR or invalid exit direction -> Not a deadlock
        if center_exit not in self.dirs:
            self.is_deadlock = False
            return False

        # 2. Check if there is an AMR on that exit arm trying to enter the intersection
        for rec in self.amr_intent_map.values():
            cur = rec.get('current_arm')
            nxt = rec.get('exit_arm')

            # If on the same arm (cur == center_exit), and
            # Its exit direction is different from the current arm (= intent to enter intersection)
            if cur == center_exit and nxt is not None and nxt != cur:
                self.is_deadlock = True
                return True

        self.is_deadlock = False
        return False
    

    def check_swap_deadlock(self):
        for rec1 in self.amr_intent_map.values():
            cur1 = rec1.get('current_arm')
            nxt1 = rec1.get('exit_arm')
            amr1_pos = rec1.get('amr_obj').pos
            amr1_next_pos = rec1.get('amr_obj').next_pos

            for rec2 in self.amr_intent_map.values():
                if rec1 == rec2:
                    continue
                cur2 = rec2.get('current_arm')
                nxt2 = rec2.get('exit_arm')
                amr2_pos = rec2.get('amr_obj').pos
                amr2_next_pos = rec2.get('amr_obj').next_pos
                rec2_tip_cell = self.target_exits.get(rec2.get('amr_obj').id)

                if (amr1_pos == rec2_tip_cell and cur1 != nxt1 and cur2 == nxt2) \
                    or (amr1_pos == amr2_next_pos and amr2_pos == amr1_next_pos):
                    self.is_deadlock = True
                    return True

        self.is_deadlock = False
        return False


    def build_prestage_paths(self):
        """
        Generate/inject pre-stage paths for deadlock resolution.
        The goal is to compress each lane from near (front) without gaps and leave the center empty.
        - If there is an AMR at the center, send it to the 'exit_arm' front first
          (if no space, send to the front of the least occupied arm (N->E->S->W tie-break)).
        - Match the path lengths of all AMRs (stationary AMRs repeat their current coordinates).
        Returns: (lanes, target_lanes, paths, max_steps)
        - lanes/target_lanes: {'N':[aid|None,...], ...} (index 0 = near/front)
        - paths: {amr_id: [(x,y), ...]} (same length)
        - max_steps: Maximum number of actions (ticks) for synchronization
        """
        cx, cy = self.center_x, self.center_y
        center = (cx, cy)
        dirs = self.dirs

        # 0) Occupancy snapshot (near->far)
        lanes = {d: [None] * len(self.lane_coords[d]) for d in dirs}
        pos2aid = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get('amr_obj')
            pos2aid[a.pos] = aid

        for d, coords in self.lane_coords.items():
            for i, p in enumerate(coords):  # i=0: near(front)
                lanes[d][i] = pos2aid.get(p, None)
        
        # --- Initial paths: Input current pos as 1st element ---
        paths = {}
        for aid, rec in self.amr_intent_map.items():
            a = rec.get('amr_obj')
            paths[aid] = [a.pos]

        # 1) Compress each arm near->far -> target_lanes
        target_lanes = {}
        for d in dirs:
            filled = [aid for aid in lanes[d] if aid is not None]  # Maintain current order
            cap = len(lanes[d])
            target_lanes[d] = filled + [None] * (cap - len(filled))

        # 2) Place center AMR (insert at front)  ✅ Modified version
        center_id = pos2aid.get(center, None)
        if center_id is not None:
            center_rec = self.amr_intent_map.get(center_id, {})
            exit_dir = center_rec.get('exit_arm')

            def occ(d): 
                return sum(1 for x in target_lanes[d] if x is not None)

            # ★ Only arms with empty spaces are candidates
            counts = {d: occ(d) for d in dirs}
            cands = [d for d in dirs if counts[d] < len(target_lanes[d])]

            host = None

            # 1) Priority to exit_dir if it has space
            if exit_dir in dirs and exit_dir in cands:
                host = exit_dir

            # 2) Otherwise, pick the arm with minimum occupancy among "arms with space (cands)" (NESW tie-break)
            elif cands:
                min_count = min(counts[d] for d in cands)
                for d in dirs:  # dirs order is tie-break
                    if d in cands and counts[d] == min_count:
                        host = d
                        break

            # 3) If host exists, incorporate into front (right shift)
            target_lanes[host] = [center_id] + target_lanes[host][:-1]                

        # -------------------------------
        # 3) lanes vs target_lanes -> Generate synchronized paths
        # -------------------------------

        # Current/Target index map (near index)
        cur_loc = {}   # aid -> (arm, idx)  (Center is ('C', None))
        for d in dirs:
            for i, aid in enumerate(lanes[d]):
                if aid is not None:
                    cur_loc[aid] = (d, i)
        if center_id is not None:
            cur_loc[center_id] = ('C', None)

        tgt_loc = {}   # aid -> (arm, idx)
        for d in dirs:
            for i, aid in enumerate(target_lanes[d]):
                if aid is not None:
                    tgt_loc[aid] = (d, i)

        # Calculate required steps for each AMR
        steps = {}
        max_steps = 0
        for aid in paths.keys():
            if aid == center_id and aid in tgt_loc:
                dist = 1  # Center -> host.front
            elif aid in cur_loc and aid in tgt_loc:
                d0, i0 = cur_loc[aid]
                d1, i1 = tgt_loc[aid]
                if d0 == d1 and d0 in dirs:
                    dist = abs(i1 - i0)  # Index difference within the same arm
                else:
                    # (This case is rare: default policy is no arm change except for center)
                    # Safely put an upper bound via center, actual coordinates are preserved below.
                    dist = 1 + (i1 if d1 in dirs else 0)
            else:
                dist = 0
            steps[aid] = dist
            if dist > max_steps:
                max_steps = dist

        # Generate per-step coordinates
        for s in range(1, max_steps + 1):
            for aid in paths.keys():
                # Initial value: last coordinate (stationary padding)
                last = paths[aid][-1]

                if aid == center_id and aid in tgt_loc:
                    # Center AMR: Jump to host.front at step 1, then fixed
                    if s == 1:
                        d1, i1 = tgt_loc[aid]
                        pos = self.lane_coords[d1][0]
                    else:
                        pos = last
                    paths[aid].append(pos)
                    continue

                # Movement within the same arm (one step at a time)
                if aid in cur_loc and aid in tgt_loc:
                    d0, i0 = cur_loc[aid]
                    d1, i1 = tgt_loc[aid]
                    if d0 == d1 and d0 in dirs:
                        m = steps[aid]
                        if m == 0:
                            paths[aid].append(last)
                            continue
                        # Direction: one step towards target
                        move_k = min(s, m)
                        # Current index = i0 + sign(i1-i0) * move_k
                        sign = 0
                        if i1 > i0: sign = 1
                        elif i1 < i0: sign = -1
                        idx = i0 + sign * move_k
                        pos = self.lane_coords[d0][idx]
                        paths[aid].append(pos)
                        continue

                # Others (stationary padding)
                paths[aid].append(last)

        self.paths = paths.copy()

        return target_lanes, paths
        
    
    def plan_action(self):
        current_stacks, targets = self.build_stacks_from_snapshot()
        dirs = list(self.dirs)
        n = len(dirs)
        dir_to_idx = {d: i for i, d in enumerate(dirs)}

        solver_input_stacks = []
        for d in dirs:
            stack_content = []
            for aid in current_stacks[d]:
                target_dir = targets.get(aid)
                if target_dir not in dir_to_idx:
                    target_dir = d
                stack_content.append(dir_to_idx[target_dir])
            solver_input_stacks.append(stack_content)

        lane_caps = [len(self.lane_coords[d]) for d in dirs]

        actions, elapsed_time, wb, hit = schedule(
            initial_stacks=solver_input_stacks,
            stack_capacities=lane_caps,
            per_stack_quota=self.stack_quota,
            order=list(range(n)),
            cache_db_path=getattr(self, "cache_db_path", None),  # Worker is read-only
            max_iters=1_000_000,
        )

        # Store only writeback to pass to main (Intersection does not access DB)
        self.cache_writeback = wb
        self.cache_hit = hit

        return actions


    def build_stacks_from_snapshot(self):
        """
        return:
            stacks  = {'N': [5, 3], 'S': [8], 'W': [2, 7]}
            targets = {5: 'E', 3: 'E', 8: 'N', 2: 'W', 7: 'S'}
        """
        dirs = self.dirs
        target_lanes, paths = self.build_prestage_paths()

        stacks = {d: [] for d in dirs}

        for d in dirs:
            near_to_far = target_lanes.get(d, [])
            ids = [aid for aid in near_to_far if aid is not None]
            for aid in reversed(ids):
                stacks[d].append(aid)
        
        targets = {}
        for aid, rec in self.amr_intent_map.items():
            tgt = rec.get('exit_arm')
            if tgt in stacks:
                targets[aid] = tgt

        return stacks, targets



    def actions_to_paths(self):
        idxs_to_dirs = {i: d for i, d in enumerate(self.dirs)}

        # 1. Obtain movement plan
        actions = self.plan_action()

        # 2. Initialization for simulation (current state)
        inter_sim, targets = self.build_stacks_from_snapshot()
        center_amr_id = None
        pending_dst = None

        # 3. Execute actions sequentially and record paths
        for src, dst in actions:
            # [Phase 1] Finalize previous action (Pending Push)
            if center_amr_id is not None and pending_dst is not None:
                # 1. Center -> Dst Push
                inter_sim[pending_dst].append(center_amr_id)

                # 2. Clear Center
                center_amr_id = None
                pending_dst = None
            
            # [Phase 2] Perform current action (Src -> Center Pull)
            # 1. Source -> Center
            mover_id = inter_sim[idxs_to_dirs[src]].pop()
            center_amr_id = mover_id
            
            # 2. Record physical position
            self._record_snapshot(inter_sim, center_amr_id=center_amr_id)

            # 3. Store destination for next step
            pending_dst = idxs_to_dirs[dst]

        # 4. Handle remaining Center AMR after loop
        if center_amr_id is not None and pending_dst is not None:
            inter_sim[pending_dst].append(center_amr_id)
            self._record_snapshot(inter_sim, center_amr_id=None)

        # 5. Path post-processing
        # Delete robots that do not pass through the center & truncate paths after passing the center
        self._post_process_paths(targets)

        return self.paths, self.target_exits, getattr(self, 'cache_writeback', None), getattr(self, 'cache_hit', False)


    def _record_snapshot(self, inter_sim, center_amr_id):
        """
        Record current coordinates for all AMRs in self.paths
        based on current inter_sim and center AMR information.
        """
        center_coord = (self.center_x, self.center_y)

        # 1. Handle AMR at the Center
        if center_amr_id is not None:
            self.paths[center_amr_id].append(center_coord)

        # 2. Handle AMRs in each lane
        for d in self.dirs:
            stack = inter_sim[d]
            lane_coords = self.lane_coords[d]

            for i, aid in enumerate(reversed(stack)):
                coords = lane_coords[i]
                self.paths[aid].append(coords)


    def _post_process_paths(self, targets):
        """
        Post-process generated self.paths:
        1. Remove AMRs that do not pass through the center (simple wait)
        2. For AMRs passing through the center:
           - Case A: Going to the original destination
               Truncate path from 'Center -> Exit' only
           - Case B: Detouring to another location
               Do not truncate path, move until the end (clear out completely)
        """
        center = (self.center_x, self.center_y)

        # Copy key list to prevent dictionary size change during iteration
        for aid in list(self.paths.keys()):
            path = self.paths[aid]

            # 1. Remove AMRs that do not pass through the center
            if center not in path and self.amr_intent_map[aid]['current_arm'] == self.amr_intent_map[aid]['exit_arm']:
                del self.paths[aid]
                continue

            # 2. Find the last point in time at the center (search from the back)
            last_center_idx = -1
            for i in range(len(path) - 1, -1, -1):
                if path[i] == center:
                    last_center_idx = i
                    break

            # --- From here, calculate exit tip ---

            # The exit direction this AMR "originally" wanted to go (set in register_amr)
            intended_target_dir = targets.get(aid)  # 'N', 'E', 'S', 'W' or None

            # --- Below: Maintain existing Case A / B truncation logic ---

            # 3. Determine actual exit direction (Actual Exit)
            #    Check coordinates of the cell immediately after the center
            if last_center_idx + 1 < len(path):
                exit_cell = path[last_center_idx + 1]
                actual_exit_dir = None

                for d, coords in self.lane_coords.items():
                    # coords[0] is usually the cell closest to the intersection (Front)
                    if exit_cell in coords:
                        actual_exit_dir = d
                        break

                # Compare destination and actual exit direction
                if (
                    actual_exit_dir is not None
                    and intended_target_dir is not None
                    and actual_exit_dir == intended_target_dir
                ):
                    # Case A: Normal exit
                    # -> Keep only up to the cell furthest from the center in the segment after the center, truncate the rest
                    max_idx = last_center_idx
                    max_dist2 = 0
                    cx, cy = center

                    for idx in range(last_center_idx + 1, len(path)):
                        x, y = path[idx]
                        dx = x - cx
                        dy = y - cy
                        dist2 = dx * dx + dy * dy
                        if dist2 > max_dist2:
                            max_dist2 = dist2
                            max_idx = idx

                    # Truncate after max_idx (include up to max_idx)
                    cut_idx = max_idx + 1
                    if cut_idx < len(path):
                        self.paths[aid] = path[:cut_idx]
                else:
                    # Case B: Detour/Avoidance -> Do not truncate
                    # Must go deep as calculated by the scheduler
                    pass
            else:
                # Case where center is the last in path (rare but possible) -> Leave as is
                pass