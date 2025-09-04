import pygame
import pyglet
import tkinter as tk
from tkinter import ttk


class GUI():
    def __init__(self, env):
        self.width_window = 3770
        self.height_window = 1040
                
        # Load simulation environment
        self.env = env
        grid = self.env.map
        height, width = grid.shape
        self.dis = min(self.width_window // width, self.height_window // height)
        self.width = self.dis * width
        self.height = self.dis * height

        
        # Main window
        self.root = tk.Tk()  
        pyglet.font.add_file('utils/D2Coding.ttf')
        self.root.title("Multi AGV System Simulator")
        self.root.resizable(False, False)
        self.root.configure(background='#000000')
        
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
        
        # AGV Algorithm Setting
        self.algorithm_label = tk.Label(self.setting, text = 'Algorithms', font = self.font_style2)
        self.algorithm_box = ttk.Combobox(self.setting, 
                                    values=["D*", "PIBT", "MADDPG"], state = 'readonly',
                                    font=self.font_style2)
        self.algorithm_box.current(0)
        self.algorithm_box.bind("<<ComboboxSelected>>", self.algorithm_changed)

        # RL Agent Setting
        self.rl_agent_var = tk.BooleanVar()
        self.rl_agent_check = tk.Checkbutton(
            self.setting,
            text="Intersection RL Agent",
            variable=self.rl_agent_var,
            font=self.font_style2,
            command=self.rl_agent_toggled
        )
        # Environment의 RL 사용 여부에 따라 초기값 설정
        self.rl_agent_var.set(getattr(self.env, 'use_rl', False))

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
        self.append_log('Multi AGV System Simulator - Hoonie_0130 (CSI Lab)')
        
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
        self.rl_agent_check.pack()
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
        self.win = pygame.display.set_mode((self.width, self.height))
        self.redrawWindow(self.env.Get_AGV())
        self.root.after(100, self.run_env())
        self.root.mainloop()
        
    # Update windows
    def redrawWindow(self, agv_list):
        pygame.display.set_caption('Warehouse Digital Twin')
        self.win.fill((32,32,32))
        self.drawMap()

        # Draw active tasks as rectangles (with AGV color)
        active_tasks = self.env.get_active_tasks()  # {agv_id: (row, col)}
        for num, (row, col) in active_tasks.items():
            color = self.env.color_map[num]
            pygame.draw.rect(
                self.win,
                color,
                (
                    int((row + 0.5) * self.dis - self.dis / 2),
                    int((col + 0.5) * self.dis - self.dis / 2),
                    int(self.dis),
                    int(self.dis)
                )
            )

        # Draw AGVs as circles
        for num, agv in agv_list.items():
            x, y = agv.pos[0], agv.pos[1]
            pygame.draw.circle(
                self.win,
                agv.color,
                (int((x + 0.5) * self.dis), int((y + 0.5) * self.dis)),
                int(self.dis / 2) - 2
            )

        # Draw goal lines if enabled
        if self.show_goal_var.get():
            for agv_id, agv in agv_list.items():
                goal_pos = self.env.controller.agv_goal.get(agv_id)
                if goal_pos:
                    start_pixel = (int((agv.pos[0] + 0.5) * self.dis), int((agv.pos[1] + 0.5) * self.dis))
                    end_pixel = (int((goal_pos[0] + 0.5) * self.dis), int((goal_pos[1] + 0.5) * self.dis))
                    pygame.draw.line(self.win, agv.color, start_pixel, end_pixel, 2)
        
        pygame.display.flip()
        
        return
    
    # Draw Map
    def drawMap(self):
        for x in range (len(self.env.map[0])):
            for y in range(len(self.env.map)):
                if self.env.map[y][x] == 1:
                    pygame.draw.rect(self.win, (160, 160, 160), (x * self.dis+1, y * self.dis+1, self.dis-2, self.dis-2))
                if self.env.map[y][x] == 6:
                    lines = self.env.find_line(x,y)
                    for line in lines:
                        pygame.draw.line(self.win, (51, 153, 255), [(x + 1/2) * self.dis, (y + 1/2) * self.dis] , [(line[0] + 1/2) * self.dis, (line[1] + 1/2) * self.dis] , 1)
                        # pygame.draw.circle(self.win, (0, 0, 255), ( (x + 1/2) * self.dis, (y + 1/2) * self.dis), self.dis / 2, 1)
                if type(self.env.map[y][x]) == str:
                    pygame.draw.rect(self.win, self.env.color.dic[self.env.map[y][x][1]], ((x + 1/2) * self.dis, (y + 1/2) * self.dis), self.dis / 2)

    # Run environment
    def run_env(self, event = None):
        if self.running_check:
            run = self.env.step(train=False)
            if run == False:
                self.running_check = False
            self.make_state_info(run)
            self.redrawWindow(self.env.Get_AGV())
        pygame.event.get()
        self.root.after(self.speed_var.get(), self.run_env)
    
    # If start button is clicked
    def start_env(self, event = None):
        self.running_check = True
        self.append_log('Start Simulation')
    
    # If stop button is clicked
    def stop_env(self, event = None):
        self.running_check = False
        self.append_log('Stop Simulation')

    # If reset button is clicked
    def reset_env(self, event = None):
        self.running_check = False
        self.env.reset()
        self.redrawWindow(self.env.Get_AGV())
        self.make_state_info(self.env.make_info())
        self.append_log('Reset Simulation') 
    
    # Append Log
    def append_log(self, msg):
        self.log_box.insert(tk.END, "{}".format(msg))
        self.log_box.update()
        self.log_box.see(tk.END)

    # Append Log
    def update_state(self, msg):
        self.state_box.insert(tk.END, "{}".format(msg))
        self.state_box.update()
        self.state_box.see(tk.END)
    
    # Clear all Log
    def clear_log(self, event = None):
        self.log_box.delete(0, self.log_box.size())
        self.log_box.see(tk.END)

    # When trajectory algorithm is changed
    def algorithm_changed(self, event):
        self.append_log("Changed Avoidance algorithm to {}".format(event.widget.get()))
        if event.widget.get() == "D*":
            self.env.controller.running_opt = 0
        if event.widget.get() == "PIBT":
            self.env.controller.running_opt = 1
        if event.widget.get() == "MADDPG":
            self.use_maddpg = True                
            
    def make_state_info(self, info_list):
        if info_list == False:
            return
        self.state_box.delete(0, self.state_box.size())
        self.update_state('{:>20} {:<10}'.format('Whole Product: ', info_list[0]))
        self.update_state('{:>20} {:<10}'.format('Throughput (/min): ', round(info_list[1], 3)))
        self.update_state(' ')
        self.update_state('{:^7} {:^7} {:^7}'.format('AGVs', 'Products', 'Mode'))
        for num, info in info_list[2].items():
            if info[1] == 0:
                self.update_state('{:^7} {:^7} {:^7}'.format(num, info[0], "Normal"))
            if info[1] == 1:
                self.update_state('{:^7} {:^7} {:^7}'.format(num, info[0], "Collision"))
            if info[1] == 2:
                self.update_state('{:^7} {:^7} {:^7}'.format(num, info[0], "Deadlock"))
        return 

    def rl_agent_toggled(self):
        if self.rl_agent_var.get():
            # RL 정책이 로드되어 있는지 확인
            if hasattr(self.env, 'rl_policy') and self.env.rl_policy is not None:
                self.env.use_rl = True
                self.append_log("Intersection RL Agent ON")
            else:
                # 정책이 없으면 경고 메시지와 함께 체크박스 해제
                self.append_log("Warning: No RL policy loaded!")
                self.rl_agent_var.set(False)
        else:
            self.env.use_rl = False
            self.append_log("Intersection RL Agent OFF")