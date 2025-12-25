import os
import json
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.AMR import AMR
from utils.Intersection import Intersection
from utils import Funct
from utils.traffic_generator import TaskSetGenerator
from utils.Controller import PIBTPlanner, CBSPlanner, BFSPlanner


def _actions_to_paths_job(iid: str, inter: "Intersection"):
    """
    서브 프로세스에서 실행할 함수.
    
    - iid: 교차로 ID (문자열)
    - inter: Intersection 객체 (해당 프로세스에서만 사용하는 복사본)
    반환:
      (iid, short_paths, target_exits)
    """
    short_paths, target_exits = inter.actions_to_paths()
    return iid, short_paths, target_exits


class ENV():
    def __init__(self, prob_path, density, max_steps, workers):
        super().__init__()
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        self.goal = set()

        self.scheduler_pool = ProcessPoolExecutor(max_workers=workers)

        self.time = 0
        
        self.map = self._load_map(map_path)
        walkable_tiles = np.count_nonzero(self.map == 0) - len(self.goal)
        num_amrs = int((walkable_tiles * density) / 100)
        print(f"\nMap width: {self.map.shape[1]}, Map height: {self.map.shape[0]}")
        print(f"Walkable tiles (value 0): {walkable_tiles}")
        print(f"Number of AMRs to spawn: {num_amrs}")
        print(f"Density: {density:.2f}%")
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
        self.planner = BFSPlanner(self.map, self.center_xs, self.center_ys)
            
        # 교차로 간 이웃 맵핑
        self.iid_neighbors: dict[str, dict[str, str]] = {
            iid: dict(inter_info.get("neighbors", {}))
            for iid, inter_info in processed_intersections.items()
        }

        # 교차로별 현재 AMR 수 (step마다 갱신)
        self.iid_inside_counts: dict[str, int] = defaultdict(int)
        
        # 모든 교차로가 동일 크기라고 가정하고, 4방향 교차로 기준 가장 짧은 3개 엣지의 합을 용량으로 설정
        v_lens = []
        h_lens = []
        for inter in processed_intersections.values():
            _, _, lN, lE, lS, lW = inter['data']
            if lN > 0: v_lens.append(lN)
            if lS > 0: v_lens.append(lS)
            if lE > 0: h_lens.append(lE)
            if lW > 0: h_lens.append(lW)            
        Lv = max(v_lens) if v_lens else 0
        Lh = max(h_lens) if h_lens else 0        
        # 4-way intersection assumed: 2 vertical arms, 2 horizontal arms
        lengths = sorted([Lv, Lv, Lh, Lh])
        self.scheduling_capacity = sum(lengths[:3])        
        print(f"Auto-configured scheduling_capacity: {self.scheduling_capacity}")

        # 각 셀이 어느 교차로에 속하는지 맵핑
        self.cell2iids: Dict[tuple[int, int], list[str]] = defaultdict(list)

        # 이벤트 셀들 (교차로 중심 + 레인 끝)
        self.event_center_cells = set()
        self.event_tip_cells = set()
        self.event_cells = set()
        self.event_cells2iid = {}

        # 맵핑 구축
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

        # 데드락 상태인 교차로
        self.deadlock_queue = []
        self.iid2sched: dict[str, set[int]] = defaultdict(set)
        self.deadlock_waiting_iids = set()

        # TaskGenerator
        self.task_generator = TaskSetGenerator(self.map, num_tasks=num_amrs, goal_positions=self.goal)

        # Color mapping
        self.color_map = Funct.Color_dict(6).dic

        self.use_scheduler = False

        self.completed_amr_steps = []

        self.completed_path_integrities: list[float] = []

        self.time_ms = []


    def reset(self):        
        self.time = 0
        self.amr_list.clear()
        
        self.task_generator.start_new_episode()

        # 모든 교차로의 내부 상태 초기화
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

        return

    
    def step(self):
        self.time += 1

        if self.task_generator.is_episode_done():
            return False

        # 1. 스케줄러 로직 (데드락 감지 및 해결)
        if self.use_scheduler:
            # (1) 교차로별 멤버 확인
            check_iids = set()
            iid2members: dict[str, list[int]] = defaultdict(list)
            stalled_iids = set()
            self.iid_inside_counts.clear()

            for amr_id, amr_obj in self.amr_list.items():
                pos = tuple(amr_obj.pos)
                
                # 현재 위치가 어떤 교차로 영역에 속하는지 확인
                if pos in self.cell2iids:
                    for iid in self.cell2iids[pos]:
                        iid2members[iid].append(amr_id)
                        self.iid_inside_counts[iid] += 1
                        
                # 데드락 체크가 필요한지 확인 (center + tip)
                if pos in self.event_cells:
                    iid = self.event_cells2iid[pos]
                    check_iids.add(iid)
            
            # 첫 스텝에서는 현 교차로, 이웃 교차로 여유 공간 카운트 초기화
            if self.time == 1:
                # 1) available_count 세팅 + 스냅샷 만들기
                resv = {}
                for iid, count in self.iid_inside_counts.items():
                    I = self.intersections[iid]
                    I.available_count = self.scheduling_capacity - count
                    resv[iid] = self.scheduling_capacity - count

                # 2) 이웃 available 스냅샷 갱신
                for iid, I in self.intersections.items():
                    neigh_map = self.iid_neighbors.get(iid, {})
                    I.neighbor_available_count = {
                        d: (resv.get(neigh_map[d], self.scheduling_capacity) if d in neigh_map else self.scheduling_capacity)
                        for d in I.dirs
                    }

            # 교차로 정지 상태인지 확인
            for iid, members in iid2members.items():
                if not members:
                    continue
                if all(self.amr_list[aid].no_move_steps >= 1 for aid in members):
                    stalled_iids.add(iid)

            # deadlock_waiting_iids도 항상 체크 대상에 포함
            check_iids |= self.deadlock_waiting_iids

            # (2-1) 잠금 해제 체크 (Deadlock 해제 시도)
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

            # 현재 스케줄 진행 중인 교차로 id 집합
            active_iids = set(self.deadlock_queue) 


            # 검증
            CAP = self.scheduling_capacity
            for iid, I in self.intersections.items():
                if iid in self.deadlock_queue:
                    continue
                # 이웃 중 스케줄링 중인 교차로가 하나라도 있으면 스킵
                if any(nid in self.deadlock_queue for nid in self.iid_neighbors.get(iid, {}).values()):
                    continue

                inside = len(iid2members.get(iid, []))   # == (등록했다면) len(I.amr_intent_map)

                # 초기 생성에서 scheduling capacity 초과면 pass
                if inside > CAP:
                    continue

                if I.available_count + inside != CAP:
                    print(f"[WARN] available mismatch iid={iid} avail={I.available_count} inside={inside} sum={I.available_count+inside} cap={CAP}")
                    return False           


            # (3-A) 데드락 체크 및 "스케줄 후보 iid" 수집
            iids_to_schedule: list[str] = []
            candidate_deadlocks: list[str] = []

            for iid in list(check_iids - active_iids):                
                I = self.intersections[iid]
                I.reset()

                # 해당 교차로 영역에 있는 모든 AMR 등록
                for amr_id in iid2members.get(iid, []):
                    amr_obj = self.amr_list[amr_id]
                    I.register_amr(amr_obj)                

                # 3-1) 데드락 여부 판단
                is_deadlock = I.check_deadlock()
                if not is_deadlock:
                    # 더 이상 데드락 아니면 pending 후보에서 제거
                    self.deadlock_waiting_iids.discard(iid)
                    continue
                
                # 3-2) 이웃 교차로가 스케줄 중이면 이번 스텝엔 스킵, 대신 pending에 등록
                if self.has_active_neighbor(iid):
                    self.deadlock_waiting_iids.add(iid)
                    continue

                # 여기까지 왔으면:
                #  - 해당 iid는 현재 데드락 상태
                #  - 이웃 active 교차로 없음 → 이번 스텝에서 스케줄 시작 가능
                self.deadlock_waiting_iids.discard(iid)
                candidate_deadlocks.append(iid)
                
            # 후보 deadlock 교차로들을 "교차로 내 AMR 수 적은 순"으로 정렬
            candidate_deadlocks.sort(
                key=lambda x: self.iid_inside_counts.get(x, 0),
                reverse=False,
            )

            # (3-A-2) 2차 패스: 정렬된 순서대로 actions_to_paths 스케줄 결정
            for iid in candidate_deadlocks:
                # 2차 패스 시점에는 앞에서 스케줄/분산이 진행되어
                # deadlock_queue가 바뀌었을 수 있으므로,
                # 이웃 active 상태를 다시 한 번 확인
                if self.has_active_neighbor(iid):
                    self.deadlock_waiting_iids.add(iid)
                    continue

                # 용량 초과 시 continue
                if self.iid_inside_counts.get(iid, 0) > self.scheduling_capacity:
                    I = self.intersections[iid]
                    self.deadlock_waiting_iids.add(iid)
                    continue

                if not self._allocate_neighbor_capacity(iid):
                    self.deadlock_waiting_iids.add(iid)
                    continue

                # 여기까지 왔으면:
                #  - 해당 iid는 데드락 상태
                #  - 이웃 active 교차로 없음 → 이번 스텝에서 스케줄 시작 가능
                self.deadlock_waiting_iids.discard(iid)

                if iid not in active_iids:
                    self.deadlock_queue.append(iid)
                    active_iids.add(iid)

                # 이 교차로에 대해 actions_to_paths 실행
                iids_to_schedule.append(iid)

            # (3-B) iids_to_schedule에 대해서 I.actions_to_paths() 병렬 실행
            futures = {}
            for iid in iids_to_schedule:
                I = self.intersections[iid]
                # 프로세스 풀에 job 제출
                fut = self.scheduler_pool.submit(_actions_to_paths_job, iid, I)
                futures[fut] = iid

            # 결과 수집 및 경로 반영
            for fut in as_completed(futures):
                iid = futures[fut]
                iid_ret, short_paths, target_exits = None, None, None
                # _actions_to_paths_job이 (iid, short_paths, target_exits)를 반환한다고 가정
                iid_ret, short_paths, target_exits = fut.result()

                # 스케줄 경로를 AMR 경로에 삽입
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

        # 2. AMR 이동
        # (A) 현재 위치 점유 맵 초기화
        current_occ = {amr.pos: amr.id for amr in self.amr_list.values()}

        # (B) 그룹 분리
        normal_amrs = [a for a in self.amr_list.values() if a.scheduling == 0]
        scheduled_amrs = [a for a in self.amr_list.values() if a.scheduling > 0]

        # -------------------------------------------------------
        # [Phase 1] 일반 로봇(Normal AMR) 먼저 이동
        # -------------------------------------------------------
        for amr in normal_amrs:
            cur_pos = amr.pos
            next_pos = amr.next_pos

            # ★ 우선순위 높은 교차로 tip 밑 scheduling capacity 초과 교차로로 진입하려는 경우 → 이 스텝에서는 대기
            if self.block_intersection(cur_pos, next_pos, normal_only=True):
                amr.no_move_steps += 1
                continue

            # 기존 충돌/점유 체크
            if next_pos not in current_occ:
                if cur_pos in current_occ and current_occ[cur_pos] == amr.id:
                    del current_occ[cur_pos]

                self._update_available_on_move_success(cur_pos, next_pos)

                amr.move()
                current_occ[amr.pos] = amr.id
            else:
                amr.no_move_steps += 1

        # -------------------------------------------------------
        # [Phase 2] 스케줄링 차단 여부 확인 (Blocking Check)
        # 일반 로봇들이 자리를 잡은 후, 스케줄링된 로봇들이 갈 수 있는지 확인
        # -------------------------------------------------------
        # 현재 일반 로봇들의 위치 집합
        normal_occ_pos = {amr.pos for amr in normal_amrs}

        # 차단된 교차로 ID 식별
        blocked_iids = set()

        for iid, members in self.iid2sched.items():
            # ✅ 순회는 snapshot(list)로, 수정은 원본 set(members)에
            for mid in list(members):
                amr = self.amr_list.get(mid, None)
                if amr is None:
                    members.discard(mid)
                    continue

                cur_pos = amr.pos
                next_pos = amr.next_pos

                # 우선순위가 높은 교차로의 끝으로 진입하려는 경우 차단
                if self.block_intersection(cur_pos, next_pos, normal_only=False):
                    blocked_iids.add(iid)
                    break

                # 다음 위치가 일반 로봇에 의해 점유된 경우 차단
                if next_pos in normal_occ_pos:
                    blocked_iids.add(iid)
                    break
        
        # -------------------------------------------------------
        # [Phase 3] 스케줄링된 로봇 이동
        # -------------------------------------------------------
        for amr in scheduled_amrs:
            # 내가 속한 스케줄이 차단되었는지 확인
            is_blocked = False
            # 역추적: 내가 어느 iid에 속해있는지 확인 (iid2sched 순회)
            for iid, members in self.iid2sched.items():
                if amr.id in members:
                    if iid in blocked_iids:
                        is_blocked = True
                    break
            
            if is_blocked:
                # 그룹 전체가 대기해야 하므로 건너뜀
                amr.no_move_steps += 1
                continue

            # 이동 수행
            if amr.pos in current_occ and current_occ[amr.pos] == amr.id:
                del current_occ[amr.pos]
            amr.move()
            current_occ[amr.pos] = amr.id


        # 3. 완료 체크 및 정보 반환
        self._check_amr_completion()

        return self.make_info()
    

    def _allocate_neighbor_capacity(self, iid):
        CAP = self.scheduling_capacity
        I = self.intersections[iid]

        dbg = getattr(self, "debug_alloc", False)

        def log(*args):
            if dbg:
                print(*args)

        # 1) 이웃 교차로별 현재 남은 자리 스냅샷
        neigh_map = self.iid_neighbors.get(iid, {})
        neigh_available = {}
        for d in "NESW":
            nid = neigh_map.get(d, None)
            if nid is None:
                neigh_available[d] = CAP
            else:
                neigh_available[d] = max(0, self.intersections[nid].available_count)

        inside_count = self.iid_inside_counts.get(iid, 0)

        # 2) 각 방향별 현재 위치 계산
        initial_need = Counter()
        for info in I.amr_intent_map.values():
            d = info.get('current_arm', None)
            if d in "NESW":
                initial_need[d] += 1

        # 3) 방향별 탈출 수요 계산
        exit_need = Counter()
        for info in I.amr_intent_map.values():
            d = info.get('exit_arm', None)
            if d in "NESW":
                exit_need[d] += 1

        # quota는 남은 여유 + 현재 그 방향 점유
        per_stack_quota = []
        for d in "NESW":
            q_total = neigh_available[d] + initial_need.get(d, 0)
            per_stack_quota.append(q_total)

        per_stack_quota = [min(q, 5) for q in per_stack_quota]
        I.stack_quota = per_stack_quota

        log(f"\n[ALLOC] iid={iid} inside={inside_count} "
            f"amr_intent={len(I.amr_intent_map)} "
            f"neigh_avail={neigh_available} quota={per_stack_quota} "
            f"initial_need={dict(initial_need)} exit_need={dict(exit_need)}")

        total_available = sum(per_stack_quota)
        if total_available < inside_count:
            log(f"[ALLOC-SKIP] iid={iid} total_available={total_available} < inside={inside_count}")
            return False

        final_need = self.predict_final_stack_lengths(exit_need, 5, per_stack_quota)
        if final_need is None:
            log(f"[ALLOC-SKIP] iid={iid} predict_final_stack_lengths returned None")
            return False

        log(f"[ALLOC] iid={iid} final_need={final_need}")

        # 4) delta 적용
        for d in "NESW":
            delta = final_need[d] - initial_need.get(d, 0)

            nid = neigh_map.get(d, None)
            if nid is None:
                I.neighbor_available_count[d] = CAP
                log(f"[ALLOC] iid={iid} dir={d} nid=None delta={delta} -> skip (neighbor_free fixed {CAP})")
                continue

            J = self.intersections[nid]
            before = J.available_count
            after = before - delta

            # 디버그용 상세 출력
            log(f"[ALLOC] iid={iid} dir={d} nid={nid} "
                f"J.avail {before} -> {after}  (delta={delta}, "
                f"final={final_need[d]}, initial={initial_need.get(d,0)})")

            J.available_count = after
            I.neighbor_available_count[d] = after

        return True
    

    def predict_final_stack_lengths(
        self,
        exit_need: Counter,          # {'N':7,'E':2,'S':2,'W':2} 같은 카운터
        stack_capacity: int = 5,     # env.stack_capacity
        per_stack_quota=None,        # [qN,qE,qS,qW] (없으면 cap로 간주)
        order="NESW",                # tie-break 순서
    ):
        # 0) 입력 정리
        idx = {d:i for i,d in enumerate(order)}
        need = [exit_need.get(d, 0) for d in order]

        cap = stack_capacity
        quota = [cap]*4 if per_stack_quota is None else [min(int(q), cap) for q in per_stack_quota]

        # 1) overflow type 계산 (총 개수 > cap)
        overflow_types = {i for i, c in enumerate(need) if c > cap}

        # 2) solved 상태에서의 "기본 길이" 가정:
        #    - overflow 타입 스택은 cap까지(가득)
        #    - 비-overflow 타입 스택은 자기 타입 개수만큼(<=cap)
        lens = [0]*4
        for i in range(4):
            lens[i] = cap if i in overflow_types else need[i]

        # 3) [1단계] 명시적 overflow 배치 룰(최소 길이, 동률 order):
        #    overflow 타입 t의 초과분(need[t]-cap)을 non-overflow 스택으로 1개씩 배치
        for t in range(4):  # N,E,S,W 순
            if t not in overflow_types:
                continue
            extra = need[t] - cap
            for _ in range(extra):
                # dst 후보: (t 자신 제외) + (overflow 스택 제외) + (cap 미만)
                cands = [j for j in range(4) if j != t and j not in overflow_types and lens[j] < cap]
                if not cands:
                    return None  # 둘 곳이 없음(구조적으로 불가능)
                min_len = min(lens[j] for j in cands)
                # 동률이면 order(N,E,S,W) 순으로
                dst = next(j for j in range(4) if j in cands and lens[j] == min_len)
                lens[dst] += 1

        # 4) [2단계] 이웃 quota 기반 overflow 배치(append_overflow_moves의 "길이" 버전):
        #    len[i] > quota[i]인 스택에서 1개 빼서, slack이 가장 큰 스택으로 이동
        while True:
            overflow_list = [(i, lens[i] - quota[i]) for i in range(4) if lens[i] > quota[i]]
            if not overflow_list:
                break

            max_over = max(k for _, k in overflow_list)
            over_srcs = [i for i, k in overflow_list if k == max_over]
            src = over_srcs[0]  # NESW 순 (index 작은 게 먼저)

            cands = [j for j in range(4) if j != src and lens[j] < quota[j] and lens[j] < cap]
            if not cands:
                # 더 이상 옮길 곳이 없으면 여기서 종료(append_overflow_moves도 break)
                break

            def slack(j):
                return (quota[j] - lens[j], cap - lens[j])

            best_sl = max(slack(j) for j in cands)
            best = [j for j in cands if slack(j) == best_sl]
            dst = best[0]  # NESW 순

            lens[src] -= 1
            lens[dst] += 1

        return {order[i]: lens[i] for i in range(4)}


    
    def _find_4cycles_from_B(self, A, B):
        """
        B를 포함하는 4-cycle들을 반환.
        cycle 형태: (B, C, D, E) => B-C-D-E-B
        """
        cycles = []
        neighbors_B = [x for x in self.iid_neighbors.get(B, {}).values() if x != A]

        # B의 두 이웃(C, E) 선택 (중복 제거 위해 i<j)
        for i in range(len(neighbors_B)):
            C = neighbors_B[i]
            for j in range(i+1, len(neighbors_B)):
                E = neighbors_B[j]

                # C와 E의 공통 이웃이 D 후보
                common = set(self.iid_neighbors.get(C, {}).values()) & set(self.iid_neighbors.get(E, {}).values())
                for D in common:
                    if D in (B, C, E):
                        continue
                    cycles.append((B, C, D, E))
        
        return cycles


    def pick_edge_cycle_for_stalled(self, A, stalled_iids, active_iids):
        """
        stalled 교차로 A에서,
        - B는 stalled가 아닌 이웃
        - B를 포함하는 4-cycle(B-C-D-E-B)이 존재
        하는 (dir_AB, B, (B,C,D,E))를 랜덤으로 하나 선택.

        반환:
        (dir_AB, B, cycle_tuple) or (None, None, None)
        """
        I = self.intersections[A]

        candidates = []
        for d, B in self.iid_neighbors.get(A, {}).items():
            if B in stalled_iids or B in active_iids:
                continue

            # A 입장에서 실제 팔이 있는 방향만
            if d not in I.dirs:
                continue

            cycles = self._find_4cycles_from_B(A, B)
            if not cycles:
                continue

            candidates.append((d, B, cycles))

        if not candidates:
            return None, None, None

        d, B, cycles = random.choice(candidates)
        cycle = random.choice(cycles)

        # 방향(시계/반시계) 랜덤 뒤집기
        if random.random() < 0.5:
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
            full_path.extend(seg[1:])  # 중복 방지
            cur = wp
        
        prefix = amr.path[:amr.path_cursor + 1]
        tail = amr.path[amr.path_cursor + 1:]

        if Bc in tail[:]:
            return  # 이미 경로에 포함된 경우 삽입 안 함

        amr.path = prefix + full_path[1:] + tail

        if amr.path_cursor + 1 < len(amr.path):
            amr.next_pos = amr.path[amr.path_cursor + 1]
        else:
            amr.next_pos = amr.pos
        
        return True
        

    def insert_scheduled_path(self, amr, short_path, target_exit, iid):
        """
        교차로 스케줄러가 생성한 경로(short_path)를 현재 AMR 경로에 삽입한다.

        - short_path : amr.pos -> 교차로 내부 merge_point 까지의 스케줄 경로
        - target_exit: 이 AMR이 '원래' 나가고 싶어하는 출구 방향 lane의 마지막 셀(tip)

        최종 경로 구성:
        prefix(지금까지 온 경로) +
        short_path[1:] (현재 위치 이후 교차로 내부 스케줄) +
        bridge(merge_point → 재합류 지점, BFS로 패치) +
        continuation(원래 AMR 경로에서 재합류 지점 이후 tail)
        """
        # 방어 코드
        if not short_path or len(short_path) < 2:
            return

        merge_point = short_path[-1]

        # 현재까지 따라온 원래 경로에서의 마지막 위치 (cursor 위치)
        if not (0 <= amr.path_cursor < len(amr.path)):
            return
        last_original_pos = amr.path[amr.path_cursor]

        # --- 1) bridge: merge_point -> rejoin_point 까지 BFS 경로 ---
        #    기본은 target_exit을 재합류 지점으로 사용,
        #    target_exit을 원래 경로에서 못 찾으면 last_original_pos로 되돌아오는 식으로 처리
        bridge = [merge_point]
        rejoin_point = None       # bridge 끝에서 원래 path로 합류할 지점
        continuation = []         # rejoin_point 이후의 원래 path tail

        # (1) target_exit 기준으로 tail을 찾으려고 시도
        exit_idx = -1
        for i in range(amr.path_cursor + 1, len(amr.path)):
            if amr.path[i] == target_exit:
                exit_idx = i
                break

        if exit_idx != -1:
            # ✅ 정상 케이스: 원래 path에 target_exit가 존재
            rejoin_point = target_exit
            continuation = amr.path[exit_idx + 1:]
        else:
            # ❌ target_exit를 원래 path에서 못 찾은 경우:
            #    last_original_pos로 되돌아가서 그 뒤 tail을 그대로 쓰기로 함
            rejoin_point = last_original_pos
            continuation = amr.path[amr.path_cursor + 1:]

        # 이 시점에서 rejoin_point는
        #   - normal case: target_exit
        #   - 예외 case: last_original_pos
        # 중 하나가 되고, continuation은 rejoin_point 이후의 tail

        # bridge 계산: merge_point -> rejoin_point
        if rejoin_point != merge_point:
            bridge = self.planner.plan_path(merge_point, rejoin_point)
        else:
            # rejoin_point가 merge_point와 같으면 bridge는 [merge_point] 그대로
            pass

        # --- 3) 새 suffix 구성 ---
        new_suffix = []

        # (a) short_path: 현재 위치는 prefix에 있으니 [1:]부터
        new_suffix.extend(short_path[1:])

        # (b) bridge: merge_point 중복 방지를 위해 [1:]부터
        if len(bridge) > 1:
            new_suffix.extend(bridge[1:])

        # (c) rejoin_point 이후 원래 경로 tail (혹은 last_original_pos 이후 tail)
        new_suffix.extend(continuation)

        # --- 4) AMR 경로/상태 업데이트 ---
        prefix = amr.path[:amr.path_cursor + 1]
        amr.path = prefix + new_suffix

        # 스케줄된 구간 길이: short_path (중복 한칸 제거)
        sched_len = len(short_path) - 1
        amr.scheduling = sched_len

        # next_pos 동기화
        if len(amr.path) > amr.path_cursor + 1:
            amr.next_pos = amr.path[amr.path_cursor + 1]
        else:
            amr.next_pos = amr.pos

    
    def has_active_neighbor(self, iid):
        """
        해당 교차로 iid의 이웃 중,
        현재 deadlock_queue에 포함된 교차로가 있는지 확인
        """
        for nid in self.iid_neighbors.get(iid, {}).values():
            if nid in self.deadlock_queue:
                return True
        return False


    def block_intersection(self, cur_pos, next_pos, normal_only=False) -> bool:
        """
        현재 위치 cur_pos에서 next_pos로 이동할 때,
        교차로 관련 정책(우선순위)에 의해 진입을 막아야 하면 True를 반환.

        우선순위(priority) 정책:
           - 현재 위치는 교차로 '밖'이거나 교차로 '중심(center)'이어야 한다.
           - 다음 위치는 어떤 교차로의 '레인 끝(tip)' 위치여야 한다.
           - 이때 진입하려는 교차로의 우선순위(= deadlock_queue 상 위치)가
             현재 교차로보다 높으면 → True (차단)

        교차로 수용량 초과 시 진입 금지(normal_only=True인 경우):
          - 진입하려는 교차로의 현재 내부 AMR 수가 스케줄링 수용량 초과 시 → True (차단)

        우선순위:
          - deadlock_queue에서 앞에 있을수록 우선순위 ↑ (index 0,1,2,...)
          - deadlock_queue에 없는 교차로는 가장 낮은 우선순위로 취급.
        """
        # 현재 위치가 교차로 중심이 아니면서 교차로 안이거나, 다음 위치가 교차로 끝이 아니면 False
        is_cur_outside = cur_pos not in self.cell2iids
        if ((cur_pos not in self.event_center_cells and not is_cur_outside) 
            or next_pos not in self.event_tip_cells):
            return False

        # 현재/다음 위치가 속한 교차로들 (교차로 중앙에 위치하면 현재 위치는 무조건 한 개의 교차로에만 속함)
        cur_iid = self.cell2iids.get(cur_pos, [])
        cur_iid_set = set(self.cell2iids.get(cur_pos, []))
        next_iid_set = set(self.cell2iids.get(next_pos, []))

        entering_iid_set = next_iid_set - cur_iid_set
        entering_iid = next(iter(entering_iid_set))

        # normal_only 모드: 교차로 스케줄링 수용량 초과 시 진입 금지
        if normal_only:
            J = self.intersections[entering_iid]
            if J.available_count <= 0:
                return True

        # 우선순위(priority) 정책   
        seq = self.deadlock_queue

        def priority(iid):
            try:
                return seq.index(iid)
            except ValueError:
                return len(seq)  # 가장 낮은 우선순위

        if is_cur_outside:
            cur_priority = len(seq)  # 가장 낮은 우선순위
        else:
            cur_priority = priority(cur_iid[0])

        # 다음 위치가 속한 교차로들 중 가장 높은 우선순위 찾기
        next_priority = priority(entering_iid)

        if next_priority < cur_priority:
            return True

        return False
    

    def _update_available_on_move_success(self, cur_pos, next_pos):
        CAP = self.scheduling_capacity

        cur_set = set(self.cell2iids.get(tuple(cur_pos), []))
        nxt_set = set(self.cell2iids.get(tuple(next_pos), []))
        entering = nxt_set - cur_set   # 새로 들어가는 교차로
        leaving  = cur_set - nxt_set   # 빠져나가는 교차로

        if not entering and not leaving:
            return

        # leaving: 자리 +1 (상한 CAP)
        for iid in leaving:
            I = self.intersections[iid]
            I.available_count = min(I.available_count + 1, CAP)

        # entering: 자리 -1 (하한 0)
        for iid in entering:
            J = self.intersections[iid]
            J.available_count = max(0, J.available_count - 1)


    def _spawn_amrs_from_task_gen(self):
        """
        [이름 변경 및 Task 모드 전용]
        TaskSetGenerator로부터 새로운 AMR을 받아 환경에 추가.
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
        [새로 추가된 함수 - Traffic 모드 전용]
        TrafficGenerator로부터 새로운 AMR을 받아 환경에 추가.
        """
        gen = self.traffic_generator
        if not gen or not gen.should_spawn_next():
            return
        
        # TrafficGenerator12는 current_time 인자가 없음
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
            
            # AMR 생성 및 등록 (amr 생성자 인자 순서 수정)
            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _direction_to_coords(self, direction, intersection_ref):
        """
        direction: 'N'|'E'|'S'|'W'
        intersection_ref: 교차로 id 문자열("x{cx}y{cy}") 또는 (cx,cy,lenN,lenE,lenS,lenW) 튜플 모두 허용
        """
        # 1) iid 문자열 → Intersection에서 스펙 가져오기
        if isinstance(intersection_ref, str):
            I = self.intersections[intersection_ref]
            # outer_entry_cells 같은 사전이 있으면 그걸 우선 사용
            if hasattr(I, "outer_entry_cells") and direction in I.outer_entry_cells:
                return I.outer_entry_cells[direction]
            center_x, center_y = I.center_x, I.center_y
            len_N, len_E, len_S, len_W = I.len_N, I.len_E, I.len_S, I.len_W

        # 2) 과거 호환: 스펙 튜플로 온 경우
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
        # 3x3 패턴들: 0=도로, 1=벽
        plus4 = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        # T자 (팔 하나 없는 방향)
        t_noN = np.array([  # 위쪽 팔 없음 (E/W/S만 열림)
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])
        t_noE = np.array([  # 오른쪽 팔 없음 (N/W/S만 열림)
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1]
        ])
        t_noS = np.array([  # 아래쪽 팔 없음 (N/E/W만 열림)
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        t_noW = np.array([  # 왼쪽 팔 없음 (N/E/S만 열림)
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 1]
        ])

        kernels = (plus4, t_noN, t_noE, t_noS, t_noW)

        # 3x3 슬라이딩 윈도우
        windows = np.lib.stride_tricks.sliding_window_view(self.map, (3, 3))
        # 각 커널에 대해 매칭 후 OR 합치기
        match_any = np.zeros(windows.shape[:2], dtype=bool)
        for K in kernels:
            match_any |= np.all(windows == K, axis=(2, 3))

        # 윈도우 좌표 → 중심 좌표(슬라이딩 오프셋 +1)
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

            # ★ 사거리/삼거리 허용: 팔이 3개 이상 존재해야 교차로 인정
            present = {d for d, L in zip("NESW", [len_N, len_E, len_S, len_W]) if L > 0}
            if len(present) >= 3:
                center_xy_to_data[(c, r)] = (c, r, len_N, len_E, len_S, len_W, present)

        processed_intersections = {}
        for (c, r), tup in center_xy_to_data.items():
            c, r, len_N, len_E, len_S, len_W, present = tup
            current_iid = f'x{c}y{r}'

            # ★ 있는 팔만 이웃 계산
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
                # ↓ 이후 단계에서 마스크/상태 0패딩에 쓰기 좋게 전달
                'present_dirs': present,
            }
        return processed_intersections

    
    def is_arm_outgoing_clear(self, iid: str, d: str) -> bool:
        I = self.intersections[iid]

        # (선택) 삼거리 대응: 존재하지 않는 팔은 금지
        present = getattr(I, "present_dirs", set(I.lane_coords.keys()))
        if d not in present:
            return False

        # 1) 해당 팔에 '바깥으로 나가려는 흐름'이 있으면 금지
        has_outgoing = bool(getattr(I, "outgoing", {}).get(d, False))
        if has_outgoing:
            return False

        # 2) 교차로 데드락인 경우 금지  ← deadlock_queue 정규화
        dq = self.deadlock_queue or []
        if dq and isinstance(dq[0], tuple):
            dead_iids = {x for (x, _) in dq}
        else:
            dead_iids = set(dq)
        if iid in dead_iids:
            return False

        # (권장) 3) 팔 팁(outer entry)이 도로이고 비어있는지 확인
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
        최근 전역 위치 시그니처를 바탕으로 정지/진동을 감지.
        True면 조기 종료해야 함.
        """
        if self.time < self._stg_min_time:
            self._sig_hist.clear()
            return False
        if not self.amr_list:
            self._sig_hist.clear()
            return False

        # 전역 시그니처: (amr_id, x, y) 튜플을 정렬한 튜플
        sig = tuple(sorted((aid, amr.pos[0], amr.pos[1]) for aid, amr in self.amr_list.items()))
        self._sig_hist.append(sig)

        # 1) 정지: 최근 N개가 모두 동일
        idle = False
        if len(self._sig_hist) >= self._stg_idle_win:
            lastN = list(self._sig_hist)[-self._stg_idle_win:]
            idle = all(s == lastN[0] for s in lastN)

        # 2) 진동: 최근 M개가 ABABAB 형태(두 개의 시그니처가 번갈아)
        osc = False
        if len(self._sig_hist) >= self._stg_osc_win:
            w = self._stg_osc_win
            lastM = list(self._sig_hist)[-w:]
            if lastM[0] != lastM[1]:
                osc = all(lastM[i] == lastM[i % 2] for i in range(w))

        if idle or osc:
            return True

        return False
    

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AMR(self):
        """GUI가 AMR 목록을 가져갈 수 있도록 하는 함수"""
        return self.amr_list

    def get_active_tasks(self):
        """GUI가 AMR의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return {amr_id: amr_obj.goal for amr_id, amr_obj in self.amr_list.items()}

    def make_info(self):
        """
        [수정] GUI에 필요한 모든 정보를 계산하여 반환합니다.
        'task' 모드와 'traffic' 모드를 명시적으로 구분하여 처리합니다.
        """
        # --- 2. 모드에 따라 통계 정보 계산 ---
        progress = self.task_generator.get_progress()
        completed_tasks = progress.get('completed_total', 0)
        total_tasks = progress.get('spawned_total', 0)

        # Success Rate 계산
        success_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # 스루풋 계산 (분 단위)
        throughput = (completed_tasks / self.time * 60) if self.time > 0 else 0.0

        active_pi = []
        for amr_obj in self.amr_list.values():
            active_pi.append(amr_obj.path_integrity_ratio())
        all_pi = self.completed_path_integrities + active_pi
        avg_pi = float(np.mean(all_pi)) if all_pi else 0.0

        avg_ms = float(np.mean(self.time_ms)) if self.time_ms else 0.0

        # --- 3. 현재 활성화된 AMR들의 상세 정보 수집 ---
        active_amr_details = {}
        for amr_id, amr_obj in self.amr_list.items():
            active_amr_details[amr_id] = {
                "steps": amr_obj.steps,
            }

        # --- 4. 최종 정보 취합하여 반환 ---
        return {
            "success_rate": success_rate,
            "throughput": throughput,
            "active_amrs": active_amr_details,
            "avg_path_integrity": avg_pi,
            "avg_inference_time": avg_ms,
            "time": self.time,
        }
