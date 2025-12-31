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

        # [Added] Zoom and panning state variables
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

        # [Added] Keyboard shortcut bindings
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

        # [Modified] Font for deadlock priority display (initialization method changed as size is dynamically adjusted based on zoom_level)
        # self.priority_font = pygame.font.Font('utils/D2Coding.ttf', int(self.dis * 0.8))
        self.font_renderer = lambda size: pygame.font.Font('utils/D2Coding.ttf', max(1, int(size)))

        self.win = pygame.display.set_mode((self.width, self.height))
        self.redrawWindow(self.env.Get_AMR())
        self.root.after(100, self.run_env())
        self.root.mainloop()

    def map_to_screen(self, map_x, map_y):
        """[Added] Convert map coordinates to screen coordinates matching current zoom/pan state"""
        screen_x = (map_x * self.zoom_level) - self.view_offset_x
        screen_y = (map_y * self.zoom_level) - self.view_offset_y
        return int(screen_x), int(screen_y)
        
    # Update Window
    def redrawWindow(self, amr_list):
        pygame.display.set_caption('Warehouse Digital Twin')
        self.win.fill((32,32,32))
        self.drawMap()

        # Display goal points (rectangles)
        active_tasks = self.env.get_active_tasks()  # {amr_id: (x,y)}
        for amr_id, (gx, gy) in reversed(list(active_tasks.items())):
            # Use env.color_map for base color
            base_color = self.env.color_map[amr_id % 6]
            sx, sy = self.map_to_screen(gx, gy)
            pygame.draw.rect(self.win, base_color, (sx, sy, self.zoom_level, self.zoom_level))

        # Display AMR
        for _, amr in amr_list.items():
            x, y = amr.pos
            sx, sy = self.map_to_screen(x + 0.5, y + 0.5)
            radius = max(1, int(self.zoom_level / 2) - 2)
            
            # [Modified] Visualization distinction based on scheduling state
            # If scheduling: thicken border or display with brighter color
            if amr.scheduling > 0:
                # Method 1: Add white border (emphasis)
                pygame.draw.circle(self.win, amr.color, (sx, sy), radius)
                # pygame.draw.circle(self.win, (255, 255, 255), (sx, sy), radius, 2) # White border
                
                # Method 2: Draw a dot in the center (indicates deadlock resolution in progress)
                pygame.draw.circle(self.win, (255, 255, 255), (sx, sy), max(1, radius // 2))
            else:
                # Normal state: basic circle
                pygame.draw.circle(self.win, amr.color, (sx, sy), radius)

        # Goal lines (optional)
        if self.show_goal_var.get():
            for amr_id, amr in amr_list.items():
                goal = amr.path[-1] if amr.path else None
                if goal:
                    start_sx, start_sy = self.map_to_screen(amr.pos[0] + 0.5, amr.pos[1] + 0.5)
                    end_sx,   end_sy   = self.map_to_screen(goal[0] + 0.5,    goal[1] + 0.5)
                    
                    # Different line color if scheduling? (optional)
                    line_color = amr.color
                    pygame.draw.line(self.win, line_color, (start_sx, start_sy), (end_sx, end_sy), 2)

        # Deadlock overlay (supports iid list or (iid, ts))
        dq = getattr(self.env, 'deadlock_queue', [])
        if dq:
            priority_colors = [(255, 0, 0), (255,165,0), (255,255,0)]
            default_color = (0, 0, 255)
            # Assume latest is added at the end -> priority from the back
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
        # [Modified] Optimized to draw only the currently visible area instead of the entire map
        grid_h, grid_w = self.env.map.shape
        
        # Calculate start/end coordinates of the map to be displayed on screen
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
        
        # [Added] Mouse event handling (zoom/pan)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.root.quit()
                return
            # Mouse wheel: zoom in/out
            if event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # Calculate map coordinates corresponding to mouse position
                map_x_before_zoom = (mouse_x + self.view_offset_x) / self.zoom_level
                map_y_before_zoom = (mouse_y + self.view_offset_y) / self.zoom_level
                
                # Change zoom level
                if event.y > 0: # Wheel up
                    self.zoom_level *= 1.1
                else: # Wheel down
                    self.zoom_level /= 1.1
                self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level))

                # After zoom, adjust offset so mouse cursor points to the same map coordinates
                self.view_offset_x = (map_x_before_zoom * self.zoom_level) - mouse_x
                self.view_offset_y = (map_y_before_zoom * self.zoom_level) - mouse_y

            # Mouse button pressed: start panning
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                self.panning = True
                self.pan_start_pos = event.pos
            
            # Mouse button released: end panning
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.panning = False

            # Mouse motion: move screen while panning
            if event.type == pygame.MOUSEMOTION and self.panning:
                dx = event.pos[0] - self.pan_start_pos[0]
                dy = event.pos[1] - self.pan_start_pos[1]
                self.view_offset_x -= dx
                self.view_offset_y -= dy
                self.pan_start_pos = event.pos

        # Redraw screen
        self.redrawWindow(self.env.Get_AMR())
        self.root.after(self.speed_var.get(), self.run_env)
    
    # If start button is clicked
    def start_env(self, event = None):
        self.running_check = True
        self.append_log('Start Simulation')
    
    def stop_env(self, event=None):
        self.running_check = False

        # --- DEBUG: Dump specific intersection ---
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

        # lane summary (length/front/tip)
        lane_summary = {}
        for d in I.dirs:
            coords = I.lane_coords.get(d, [])
            lane_summary[d] = {
                "len": len(coords),
                "front": coords[0] if coords else None,   # cell closest to center
                "tip": coords[-1] if coords else None,    # arm tip
            }

        # intent summary
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
            pass  # If aid types are mixed, maintain original order

        # Assume stack_quota corresponds to I.dirs order (output as raw if lengths differ)
        if isinstance(I.stack_quota, (list, tuple)) and len(I.stack_quota) == len(I.dirs):
            quota_map = {d: I.stack_quota[i] for i, d in enumerate(I.dirs)}
        else:
            quota_map = {"raw": I.stack_quota}

        # env-level meta
        inside_count = (getattr(env, "iid_inside_counts", {}) or {}).get(iid, None)
        neigh_map = (getattr(env, "iid_neighbors", {}) or {}).get(iid, {}) or {}
        scheduled_set = (getattr(env, "iid2sched", {}) or {}).get(iid, set()) or set()
        dq = getattr(env, "deadlock_queue", []) or []
        in_deadlock_queue = any(
            (x == iid) or (isinstance(x, (tuple, list)) and len(x) > 0 and x[0] == iid)
            for x in dq
        )

        # Also include neighbor intersection snapshots
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

        # paths/target_exits can be too long, so only samples
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

        # GUI log only one-line summary
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

    # [Added] Space bar toggle function
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
        # [Modified] Commented out the line below to prevent auto-scrolling
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
        [Modified] Update State Panel according to the new info_dict structure.
        """
        if not isinstance(info_dict, dict):
            return
        
        self.state_box.delete(0, self.state_box.size())

        # --- Top summary information ---
        time = info_dict.get("time", 0)
        success_rate = info_dict.get("success_rate", 0.0)
        throughput = info_dict.get("throughput", 0.0)
        avg_integrity = info_dict.get("avg_path_integrity", 0.0)

        self.update_state('{:>20} {:<10}'.format('Time Step: ', time))
        self.update_state('{:>20} {:<10.2%}'.format('Success Rate: ', success_rate))
        self.update_state('{:>20} {:<10.2f}'.format('Throughput (/min): ', throughput))
        self.update_state('{:>20} {:<10.2f}'.format('Avg PI: ', avg_integrity))
        self.update_state(' ')

        # --- Individual AMR information ---
        active_amrs = info_dict.get("active_amrs", {})
        self.update_state('{:^10} {:^10}'.format('AMR ID', 'Steps'))
        self.update_state('-' * 40)

        # Sort AMR IDs by converting to integers
        sorted_amr_ids = sorted(active_amrs.keys(), key=int)

        for amr_id in sorted_amr_ids:
            details = active_amrs[amr_id]
            steps = details.get("steps", 0)
            self.update_state('{:^10} {:^10}'.format(amr_id, steps))
            
        return 

    def scheduler_toggled(self):
        self.env.use_scheduler = bool(self.scheduler_var.get())
        self.append_log(f"Scheduler {'ON' if self.env.use_scheduler else 'OFF'}")