from distutils.log import fatal
from select import select
from colorama import Back
import pygame
import pyglet
import tkinter as tk
from tkinter import ttk
import os
import platform
from tkinter import font
import threading
from tkinter import filedialog
import torch

import Environment
from agent import Actor

class GUI():
    def __init__(self, env):
        self.rows = 100
        self.width = 1000
        self.height = 1000
        self.dis = self.width // self.rows
                
        # Load simulation environment
        self.env = env
        
        # Main window
        self.root = tk.Tk()  
        pyglet.font.add_file('D2Coding.ttf')
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
        self.algorithm_label = tk.Label(self.setting, text = 'DAA Algorithms', font = self.font_style2)
        self.algorithm_box = ttk.Combobox(self.setting, 
                                    values=["Not Used", "Yoo (2005)", "Moorthy (2003)", "Kim (2007)", "MADDPG"], state = 'readonly',
                                    font=self.font_style2)
        self.algorithm_box.current(0)
        self.algorithm_box.bind("<<ComboboxSelected>>", self.algorithm_changed)
        
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
        self.setting.pack_propagate(0)
        
        self.state.pack()
        self.state_label.pack()
        self.state_box.pack()
        self.state.pack_propagate(0)
        
        self.log.pack()
        self.log_label.pack()
        self.log_box.pack()
        self.log.pack_propagate(0)

        # Load MADDPG model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actors = {agent: Actor(obs_dim=24, act_dim=5) for agent in self.env.agv_list.keys()}
        for agent in self.actors:
            self.actors[agent].load_state_dict(torch.load(f"./checkpoints/episode_1/actor_{agent}.pth", map_location='cpu'))
            self.actors[agent].eval()        
        
        # Start pygame
        pygame.init()
        self.win = pygame.display.set_mode((self.width, self.height))
        self.redrawWindow(self.env.Get_AGV())
        self.root.after(1000, self.run_env())
        self.root.mainloop()
        
    # Update windows
    def redrawWindow(self, agv_list):
        # self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Warehouse Digital Twin')
        self.win.fill((32,32,32))
        
        # Draw AGVs !
        for num, agv in agv_list.items():
            x, y = agv.pos[0], agv.pos[1]
            pygame.draw.rect(self.win, agv.color, (x * self.dis +1, y * self.dis+1, self.dis - 2, self.dis - 2))
        
        # Draw Maps !
        self.drawMap()
            
        # pygame.display.update()
        pygame.display.flip()
        return
    
    # Draw Map
    def drawMap(self):
        for x in range (100):
            for y in range(100):
                if self.env.map[y][x] == 1:
                    pygame.draw.rect(self.win, (160, 160, 160), (x * self.dis+1, y * self.dis+1, self.dis-2, self.dis-2))
                if self.env.map[y][x] == 6:
                    lines = self.env.find_line(x,y)
                    for line in lines:
                        pygame.draw.line(self.win, (51, 153, 255), [(x + 1/2) * self.dis, (y + 1/2) * self.dis] , [(line[0] + 1/2) * self.dis, (line[1] + 1/2) * self.dis] , 1)
                        # pygame.draw.circle(self.win, (0, 0, 255), ( (x + 1/2) * self.dis, (y + 1/2) * self.dis), self.dis / 2, 1)
                if type(self.env.map[y][x]) == str:
                    pygame.draw.circle(self.win, self.env.color.dic[self.env.map[y][x][1]], ( (x + 1/2) * self.dis, (y + 1/2) * self.dis), self.dis / 2)

    # Run environment
    def run_env(self, event=None):
        if self.running_check:
            if hasattr(self, 'use_maddpg') and self.use_maddpg:
                actions = []
                for agent_id, agent in self.env.agv_list.items():
                    state = self.env.get_state(agent_id)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    state_tensor = state_tensor.to(next(self.actors[agent_id].parameters()).device)

                    action_logits = self.actors[agent_id](state_tensor).squeeze(0)

                    action_mask = self.env.valid_actions(int(state[0]), int(state[1]))
                    action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=state_tensor.device)

                    masked_logits = action_logits + (1 - action_mask_tensor) * (-1e9)

                    action_probs = torch.softmax(masked_logits, dim=-1)
                    action = torch.multinomial(action_probs, 1).item()

                    actions.append(action)

                run = self.env.demo_step(actions)
            else:
                run = self.env.Run()

            if run == False:
                self.running_check = False

            self.make_state_info(run)
            self.redrawWindow(self.env.Get_AGV())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

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
        self.env = Environment.ENV()
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
        if event.widget.get() == "Not Used":
            self.env.controller.running_opt = 0
            self.use_maddpg = False
        if event.widget.get() == "Yoo (2005)":
            self.env.controller.running_opt = 1
            self.use_maddpg = False
        if event.widget.get() == "Moorthy (2003)":
            self.env.controller.running_opt = 2
            self.use_maddpg = False
        if event.widget.get() == "Kim (2007)":
            self.env.controller.running_opt = 3
            self.use_maddpg = False
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
                self.update_state('{:^7} {:^7} {:^7}'.format(num, info[0], "Deadlock"))
        return 
