import pygame
import pyglet
import tkinter as tk
from tkinter import ttk


class GUI():
    def __init__(self, env):
        self.width_window = 1880
        self.height_window = 1020
                
        # Load simulation environment
        self.env = env
        grid = self.env.map
        height, width = grid.shape
        initial_dis = min(self.width_window // width, self.height_window // height)
        self.width = initial_dis * width
        self.height = initial_dis * height

        # [추가] 확대/축소 및 패닝 상태 변수
        self.zoom_level = float(initial_dis)
        self.min_zoom = 0.5
        self.max_zoom = 50.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.panning = False
        self.pan_start_pos = (0, 0)
        
        # Main window
        self.root = tk.Tk()  
        pyglet.font.add_file('utils/D2Coding.ttf')
        self.root.title("Multi AMR System Simulator")
        self.root.resizable(True, True)
        self.root.configure(background='#000000')

        # [추가] 키보드 단축키 바인딩
        self.root.bind("<space>", self.toggle_run)
        self.root.bind("<r>", self.reset_env)
        self.root.bind("<R>", self.reset_env)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        
        # IF GUI mode is running
        self.running_check = False
        
        # font option
        self.root.option_add('*Dialog.msg.font', 'D2Coding Nerd Font 12')
        self.font_style1 = ('D2Coding Nerd Font', 14)
        self.font_style2 = ('D2Coding Nerd Font', 10)
        
        # Large Frame
        # self.win_frame = tk.Frame(self.root, width = self.width + 300, height = self.height, 
        #                        highlightbackground = '#595959', highlightthickness = 2)
        self.win_frame = tk.Frame(self.root, width = 800, height = 500, 
                                highlightbackground = '#595959', highlightthickness = 2)

        # menu (left side)
        self.menu = tk.Frame(self.win_frame, width = 200, height = 516, highlightbackground = '#595959', highlightthickness=2)
        self.menu_label = tk.Label(self.menu, text = 'Control Panel', font = self.font_style1)
        self.Start_button = tk.Button(self.menu, text= "Start", bg = '#728f96', 
                                    font = self.font_style1, activebackground='#d45f5f')
        self.Start_button.bind("<Button-1>", self.start_env)
        
        self.Stop_button = tk.Button(self.menu, text= "Stop", bg = '#728f96', 
                                    font = self.font_style1, activebackground='#d45f5f')
        self.Stop_button.bind("<Button-1>", self.stop_env)
        
        self.Reset_button = tk.Button(self.menu, text = "Reset", font = self.font_style1, 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Reset_button.bind("<Button-1>", self.reset_env)
        
        self.Clear_button = tk.Button(self.menu, text = "Clear Log", font = self.font_style1, 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Clear_button.bind("<Button-1>", self.clear_log)
        
        # Setting(Middle side)
        self.setting = tk.Frame(self.win_frame, width = 200, height = 516, highlightbackground = '#595959', highlightthickness=2)   
        self.setting_label = tk.Label(self.setting, text = 'Setting Panel', font = self.font_style1)   
        
        # Speed setting
        self.speed_var = tk.IntVar()
        self.speed_label = tk.Label(self.setting, text = 'Simulation Speed', font = self.font_style2)
        self.speed_scale = tk.Scale(self.setting, variable = self.speed_var, orient="horizontal", state = 'active',
                                    showvalue = True, from_ = 1000, to = 1, length = 200,
                                    highlightbackground = '#728f96', activebackground = '#728f96', font=self.font_style2)
        self.speed_scale.set(100)
        
        # AMR Algorithm Setting
        self.algorithm_label = tk.Label(self.setting, text = 'Algorithms', font = self.font_style2)
        self.algorithm_box = ttk.Combobox(self.setting, 
                                    values=["BFS", "CBS"], state = 'readonly',
                                    font=self.font_style2)
        self.algorithm_box.current(0)
        self.algorithm_box.bind("<<ComboboxSelected>>", self.algorithm_changed)

        self.scheduler_var = tk.BooleanVar()
        self.scheduler_check = tk.Checkbutton(
            self.setting,
            text="Use Scheduler",
            variable=self.scheduler_var,
            font=self.font_style2,
            command=self.scheduler_toggled
        )
        self.scheduler_var.set(getattr(self.env, 'use_scheduler', False))

        # Show Goal Line Setting
        self.show_goal_var = tk.BooleanVar()
        self.show_goal_check = tk.Checkbutton(
            self.setting,
            text="Show Goal Lines",
            variable=self.show_goal_var,
            font=self.font_style2
        )
        self.show_goal_var.set(False)
        
        # State (Right side)
        self.state = tk.Frame(self.win_frame, width = 400, height = 350, highlightbackground = '#595959', highlightthickness=2)   
        self.state_label = tk.Label(self.state, text = 'State Panel', font = self.font_style1)  
        
        self.state_scroll = tk.Scrollbar(self.state, orient='vertical')
        self.state_box = tk.Listbox(self.state, yscrollcommand = self.state_scroll.set, width = 400, height = 400, font = self.font_style2)
        self.state_scroll.config(command=self.state_box.yview)
        
        # Log (Right side)
        self.log = tk.Frame(self.win_frame, width = 400, height = 166, highlightbackground = '#595959', highlightthickness=2)   
        self.log_label = tk.Label(self.log, text = 'Log Panel', font = self.font_style1) 
        self.log_scroll = tk.Scrollbar(self.log, orient='vertical')
        self.log_box = tk.Listbox(self.log, yscrollcommand = self.log_scroll.set, width = 400, height = 400, font=self.font_style2)
        self.log_scroll.config(command=self.log_box.yview)
        
        # Start log
        self.append_log('Multi AMR System Simulator - Hoonie_0130 (CSI Lab)')
        
        # pygame
        self.pygame_frame = tk.Frame(self.win_frame, width = self.width, height = self.height, 
                                    highlightbackground='#595959', highlightthickness=2)
        self.embed = tk.Frame(self.pygame_frame, width = self.width, height = self.height)

        # Packing
        self.win_frame.pack(expand = True)
        self.win_frame.pack_propagate(0)

        self.menu.pack(side="left")
        self.menu.pack_propagate(0)
        self.menu_label.pack()
        
        self.Start_button.pack(ipadx = 60)
        self.Stop_button.pack(ipadx = 60)
        self.Reset_button.pack(ipadx = 60)
        self.Clear_button.pack(ipadx= 60)
        
        self.setting.pack(side = "left", anchor = 'n')
        self.setting_label.pack()
        self.speed_label.pack()
        self.speed_scale.pack()
        self.algorithm_label.pack()
        self.algorithm_box.pack()
        self.scheduler_check.pack()
        self.show_goal_check.pack()
        self.setting.pack_propagate(0)
        
        self.state.pack()
        self.state_label.pack()
        self.state_box.pack()
        self.state.pack_propagate(0)
        
        self.log.pack()
        self.log_label.pack()
        self.log_box.pack()
        self.log.pack_propagate(0)      
        
        # Start pygame
        pygame.init()

        # [수정] 데드락 우선순위 표시용 폰트 (zoom_level에 따라 동적으로 크기 조절되므로 초기화 방식 변경)
        # self.priority_font = pygame.font.Font('utils/D2Coding.ttf', int(self.dis * 0.8))
        self.font_renderer = lambda size: pygame.font.Font('utils/D2Coding.ttf', max(1, int(size)))

        self.win = pygame.display.set_mode((self.width, self.height))
        self.redrawWindow(self.env.Get_AMR())
        self.root.after(100, self.run_env())
        self.root.mainloop()

    def map_to_screen(self, map_x, map_y):
        """[추가] 맵 좌표를 현재 zoom/pan 상태에 맞는 화면 좌표로 변환"""
        screen_x = (map_x * self.zoom_level) - self.view_offset_x
        screen_y = (map_y * self.zoom_level) - self.view_offset_y
        return int(screen_x), int(screen_y)
        
    # Update Window
    def redrawWindow(self, amr_list):
        pygame.display.set_caption('Warehouse Digital Twin')
        self.win.fill((32,32,32))
        self.drawMap()

        # 목표 지점 표시(사각형)
        active_tasks = self.env.get_active_tasks()  # {amr_id: (x,y)}
        for amr_id, (gx, gy) in reversed(list(active_tasks.items())):
            # 기본 색상은 env.color_map 사용
            base_color = self.env.color_map[amr_id % 6]
            sx, sy = self.map_to_screen(gx, gy)
            pygame.draw.rect(self.win, base_color, (sx, sy, self.zoom_level, self.zoom_level))

        # AMR 표시
        for _, amr in amr_list.items():
            x, y = amr.pos
            sx, sy = self.map_to_screen(x + 0.5, y + 0.5)
            radius = max(1, int(self.zoom_level / 2) - 2)
            
            # [수정] 스케줄링 상태에 따른 시각화 구분
            # 스케줄링 중이면: 테두리를 두껍게 하거나, 색상을 밝게 표시
            if amr.scheduling > 0:
                # 방법 1: 흰색 테두리 추가 (강조)
                pygame.draw.circle(self.win, amr.color, (sx, sy), radius)
                # pygame.draw.circle(self.win, (255, 255, 255), (sx, sy), radius, 2) # 흰색 테두리
                
                # 방법 2: 중앙에 점 찍기 (Deadlock 해결 중임을 표시)
                pygame.draw.circle(self.win, (255, 255, 255), (sx, sy), max(1, radius // 2))
            else:
                # 일반 상태: 기본 원
                pygame.draw.circle(self.win, amr.color, (sx, sy), radius)

        # 목표 라인 (옵션)
        if self.show_goal_var.get():
            for amr_id, amr in amr_list.items():
                goal = amr.path[-1] if amr.path else None
                if goal:
                    start_sx, start_sy = self.map_to_screen(amr.pos[0] + 0.5, amr.pos[1] + 0.5)
                    end_sx,   end_sy   = self.map_to_screen(goal[0] + 0.5,    goal[1] + 0.5)
                    
                    # 라인 색상도 스케줄링 중이면 다르게? (선택사항)
                    line_color = amr.color
                    pygame.draw.line(self.win, line_color, (start_sx, start_sy), (end_sx, end_sy), 2)

        # Deadlock overlay (iid 리스트 또는 (iid, ts) 지원)
        dq = getattr(self.env, 'deadlock_queue', [])
        if dq:
            priority_colors = [(255, 0, 0), (255,165,0), (255,255,0)]
            default_color = (0, 0, 255)
            # 최신이 뒤에 쌓인다고 가정 → 우선순위는 뒤에서부터
            for priority, entry in enumerate(reversed(dq)):
                iid = entry[0] if isinstance(entry, (tuple, list)) else entry
                I = self.env.intersections.get(iid)
                if not I:
                    continue
                x_min, x_max = I.center_x - I.len_W, I.center_x + I.len_E
                y_min, y_max = I.center_y - I.len_N, I.center_y + I.len_S
                px, py = self.map_to_screen(x_min, y_min)
                p_width  = (x_max - x_min + 1) * self.zoom_level
                p_height = (y_max - y_min + 1) * self.zoom_level
                box_color = priority_colors[priority] if priority < len(priority_colors) else default_color
                pygame.draw.rect(self.win, box_color, (px, py, p_width, p_height), 3)

                font = self.font_renderer(self.zoom_level * 0.8)
                text_surface = font.render(str(iid), True, (255,255,0))
                self.win.blit(text_surface, (px + 5, py + 5))        

        pygame.display.flip()

    
    # Draw Map
    def drawMap(self):
        # [수정] 전체 맵을 그리는 대신, 현재 보이는 영역만 그리도록 최적화
        grid_h, grid_w = self.env.map.shape
        
        # 화면에 보일 맵의 시작/끝 좌표 계산
        start_col = max(0, int(self.view_offset_x / self.zoom_level))
        end_col = min(grid_w, int((self.view_offset_x + self.width) / self.zoom_level) + 1)
        start_row = max(0, int(self.view_offset_y / self.zoom_level))
        end_row = min(grid_h, int((self.view_offset_y + self.height) / self.zoom_level) + 1)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                sx, sy = self.map_to_screen(x, y)
                if self.env.map[y][x] == 1:
                    pygame.draw.rect(self.win, (160, 160, 160), (sx + 1, sy + 1, self.zoom_level - 2, self.zoom_level - 2))

    # Run environment
    def run_env(self, event = None):
        if self.running_check:
            run = self.env.step()
            if run == False:
                self.running_check = False
            self.make_state_info(run)
        
        # [추가] 마우스 이벤트 처리 (확대/축소 및 패닝)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.root.quit()
                return
            # 마우스 휠: 확대/축소
            if event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # 마우스 위치에 해당하는 맵 좌표 계산
                map_x_before_zoom = (mouse_x + self.view_offset_x) / self.zoom_level
                map_y_before_zoom = (mouse_y + self.view_offset_y) / self.zoom_level
                
                # 줌 레벨 변경
                if event.y > 0: # 휠 위로
                    self.zoom_level *= 1.1
                else: # 휠 아래로
                    self.zoom_level /= 1.1
                self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level))

                # 줌 이후, 마우스 커서가 동일한 맵 좌표를 가리키도록 오프셋 조정
                self.view_offset_x = (map_x_before_zoom * self.zoom_level) - mouse_x
                self.view_offset_y = (map_y_before_zoom * self.zoom_level) - mouse_y

            # 마우스 버튼 누름: 패닝 시작
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # 좌클릭
                self.panning = True
                self.pan_start_pos = event.pos
            
            # 마우스 버튼 뗌: 패닝 종료
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.panning = False

            # 마우스 이동: 패닝 중일 때 화면 이동
            if event.type == pygame.MOUSEMOTION and self.panning:
                dx = event.pos[0] - self.pan_start_pos[0]
                dy = event.pos[1] - self.pan_start_pos[1]
                self.view_offset_x -= dx
                self.view_offset_y -= dy
                self.pan_start_pos = event.pos

        # 화면 다시 그리기
        self.redrawWindow(self.env.Get_AMR())
        self.root.after(self.speed_var.get(), self.run_env)
    
    # If start button is clicked
    def start_env(self, event = None):
        self.running_check = True
        self.append_log('Start Simulation')
    
    def stop_env(self, event=None):
        self.running_check = False

        # --- DEBUG: 특정 교차로 덤프 ---
        iid = "x108y108"
        env = getattr(self, "env", None)

        if env is None:
            print("[DEBUG] stop_env: self.env is None")
            self.append_log('Stop Simulation')
            return

        inters = getattr(env, "intersections", {}) or {}
        I = inters.get(iid, None)

        print(I.check_deadlock())

        if I is None:
            print(f"[DEBUG] stop_env: intersection not found: {iid} (known={len(inters)})")
            self.append_log('Stop Simulation')
            return

        from collections import Counter
        import pprint

        # lane 요약 (길이/앞칸/팁)
        lane_summary = {}
        for d in I.dirs:
            coords = I.lane_coords.get(d, [])
            lane_summary[d] = {
                "len": len(coords),
                "front": coords[0] if coords else None,   # center에 가장 가까운 칸
                "tip": coords[-1] if coords else None,    # 팔 끝 tip
            }

        # intent 요약
        amr_map = I.amr_intent_map or {}
        cur_cnt = Counter()
        exit_cnt = Counter()

        intent_rows = []
        for aid, rec in amr_map.items():
            cur = rec.get("current_arm")
            nxt = rec.get("exit_arm")
            cur_cnt[cur] += 1
            exit_cnt[nxt] += 1

            a = rec.get("amr_obj")
            intent_rows.append({
                "aid": aid,
                "pos": getattr(a, "pos", None),
                "path_cursor": getattr(a, "path_cursor", None),
                "cur": cur,
                "exit": nxt,
                "target_tip": (I.target_exits.get(aid) if isinstance(I.target_exits, dict) else None),
            })
        try:
            intent_rows.sort(key=lambda r: r["aid"])
        except Exception:
            pass  # aid 타입이 섞여있으면 그냥 원래 순서 유지

        # stack_quota는 I.dirs 순서에 대응한다고 가정 (길이 다르면 raw로 출력)
        if isinstance(I.stack_quota, (list, tuple)) and len(I.stack_quota) == len(I.dirs):
            quota_map = {d: I.stack_quota[i] for i, d in enumerate(I.dirs)}
        else:
            quota_map = {"raw": I.stack_quota}

        # env-level 메타
        inside_count = (getattr(env, "iid_inside_counts", {}) or {}).get(iid, None)
        neigh_map = (getattr(env, "iid_neighbors", {}) or {}).get(iid, {}) or {}
        scheduled_set = (getattr(env, "iid2sched", {}) or {}).get(iid, set()) or set()
        dq = getattr(env, "deadlock_queue", []) or []
        in_deadlock_queue = any(
            (x == iid) or (isinstance(x, (tuple, list)) and len(x) > 0 and x[0] == iid)
            for x in dq
        )

        # 이웃 교차로 스냅샷도 같이
        neighbor_snap = {}
        for d, nid in neigh_map.items():
            J = inters.get(nid)
            neighbor_snap[d] = {
                "iid": nid,
                "exists": J is not None,
                "inside": (getattr(env, "iid_inside_counts", {}) or {}).get(nid, None),
                "avail": getattr(J, "available_count", None) if J else None,
                "cap": getattr(J, "scheduling_capacity", None) if J else None,
                "deadlock": getattr(J, "is_deadlock", None) if J else None,
                "intents": len(getattr(J, "amr_intent_map", {}) or {}) if J else None,
            }

        # paths/target_exits는 너무 길어질 수 있어서 샘플만
        paths = I.paths or {}
        try:
            paths_len_sample = {aid: len(p) for aid, p in list(paths.items())[:20]}
        except Exception:
            paths_len_sample = {}

        target_exits = I.target_exits or {}
        try:
            target_exits_sample = dict(list(target_exits.items())[:20]) if isinstance(target_exits, dict) else target_exits
        except Exception:
            target_exits_sample = target_exits

        dump = {
            "iid": I.id,
            "center": (I.center_x, I.center_y),
            "arm_lengths": {"N": I.len_N, "E": I.len_E, "S": I.len_S, "W": I.len_W},
            "present_dirs": sorted(list(I.present_dirs)),
            "dirs": list(I.dirs),
            "lane_summary": lane_summary,

            "scheduling_capacity": I.scheduling_capacity,
            "available_count": I.available_count,
            "neighbor_available_count": dict(I.neighbor_available_count or {}),
            "stack_quota": quota_map,

            "is_deadlock": I.is_deadlock,
            "env_inside_count": inside_count,
            "env_scheduled_members_count": len(scheduled_set),
            "env_scheduled_members_sample": sorted(list(scheduled_set))[:50],
            "in_deadlock_queue": in_deadlock_queue,

            "env_neighbors": neigh_map,
            "neighbor_snapshot": neighbor_snap,

            "amr_intent_count": len(amr_map),
            "amr_intent_current_arm_counts": dict(cur_cnt),
            "amr_intent_exit_arm_counts": dict(exit_cnt),
            "amr_intent_sample": intent_rows[:20],

            "paths_count": len(paths),
            "paths_len_sample": paths_len_sample,
            "target_exits_sample": target_exits_sample,
        }

        print("\n" + "=" * 90)
        print(f"[DEBUG] stop_env() intersection dump: {iid}")
        print(pprint.pformat(dump, width=160, sort_dicts=False))
        print("=" * 90 + "\n")

        # GUI 로그에는 한 줄 요약만
        try:
            self.append_log(
                f"[DEBUG] {iid} inside={inside_count} "
                f"avail={I.available_count}/{I.scheduling_capacity} "
                f"intents={len(amr_map)} deadlock={I.is_deadlock}"
            )
        except Exception:
            pass

        self.append_log('Stop Simulation')

    # If reset button is clicked
    def reset_env(self, event = None):
        self.running_check = False
        self.env.reset()
        self.redrawWindow(self.env.Get_AMR())
        self.make_state_info(self.env.make_info())
        self.append_log('Reset Simulation') 

    # [추가] Space 바 토글 기능
    def toggle_run(self, event=None):
        if self.running_check:
            self.stop_env(event)
        else:
            self.start_env(event)
    
    # Append Log
    def append_log(self, msg):
        self.log_box.insert(tk.END, "{}".format(msg))
        self.log_box.update()
        self.log_box.see(tk.END)

    # Append Log
    def update_state(self, msg):
        self.state_box.insert(tk.END, "{}".format(msg))
        self.state_box.update()
        # [수정] 자동 스크롤을 방지하기 위해 아래 줄을 주석 처리
        # self.state_box.see(tk.END)
    
    # Clear all Log
    def clear_log(self, event = None):
        self.log_box.delete(0, self.log_box.size())
        self.log_box.see(tk.END)

    # When trajectory algorithm is changed
    def algorithm_changed(self, event):
        self.append_log("Changed Avoidance algorithm to {}".format(event.widget.get()))
        if event.widget.get() == "BFS":
            self.env.controller.running_opt = 0
        if event.widget.get() == "CBS":
            self.env.controller.running_opt = 1

    def make_state_info(self, info_dict):
        """
        [수정] 새로운 info_dict 구조에 맞춰 State Panel을 업데이트합니다.
        """
        if not isinstance(info_dict, dict):
            return
        
        self.state_box.delete(0, self.state_box.size())

        # --- 상단 요약 정보 ---
        time = info_dict.get("time", 0)
        success_rate = info_dict.get("success_rate", 0.0)
        throughput = info_dict.get("throughput", 0.0)
        avg_integrity = info_dict.get("avg_path_integrity", 0.0)

        self.update_state('{:>20} {:<10}'.format('Time Step: ', time))
        self.update_state('{:>20} {:<10.2%}'.format('Success Rate: ', success_rate))
        self.update_state('{:>20} {:<10.2f}'.format('Throughput (/min): ', throughput))
        self.update_state('{:>20} {:<10.2f}'.format('Avg PI: ', avg_integrity))
        self.update_state(' ')

        # --- 개별 AMR 정보 ---
        active_amrs = info_dict.get("active_amrs", {})
        self.update_state('{:^10} {:^10}'.format('AMR ID', 'Steps'))
        self.update_state('-' * 40)

        # AMR ID를 정수로 변환하여 정렬
        sorted_amr_ids = sorted(active_amrs.keys(), key=int)

        for amr_id in sorted_amr_ids:
            details = active_amrs[amr_id]
            steps = details.get("steps", 0)
            self.update_state('{:^10} {:^10}'.format(amr_id, steps))
            
        return 

    def scheduler_toggled(self):
        self.env.use_scheduler = bool(self.scheduler_var.get())
        self.append_log(f"Scheduler {'ON' if self.env.use_scheduler else 'OFF'}")