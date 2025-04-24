import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import Funct
from map import map


class Controller:
    def __init__(self):
        self.map = map

    def graphing(self):
        self.graph = {}
        for x in range (100):
            for y in range(100):
                # normal node 
                if self.map[y][x] == 6:
                    neighbors = self.find_neighbors(x,y)
                    nodes = {}
                    for neighbor in neighbors:
                        nodes[neighbor] = Funct.get_distance((x,y), neighbor)
                    self.graph[(x,y)] = nodes
                    
                if type(self.map[y][x]) == str:
                    neighbors = self.find_neighbors(x,y, False)
                    nodes = {}
                    for neighbor in neighbors:
                        nodes[neighbor] = Funct.get_distance((x,y), neighbor)
                    self.graph[(x,y)] = nodes   
    
    
    def find_neighbors(self, x, y, rout = True):
        line_list = []
        distance = 0
        poss_x = x
        poss_y = y
        # up
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_x += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
                
        distance = 0
        poss_x = x
        poss_y = y
        
        # down
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_x -= 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
        
        distance = 0
        poss_x = x
        poss_y = y
        
        # right
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
            if (self.map[poss_y][poss_x] == 1):
                break
            
        distance = 0
        poss_x = x
        poss_y = y
        
        # left
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y -= 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
            
        return line_list
    
    def visualize_grid_matplotlib(self, agv_pos=None, goal_pos=None):
        height = len(self.map)
        width = len(self.map[0])

        # 1. 이동 가능한 셀들 계산 (노드 + 엣지 포함)
        white_cells = set(self.graph.keys())  # 노드
        for start, neighbors in self.graph.items():
            for end in neighbors:
                # 엣지 경로상의 셀들을 모두 포함
                x0, y0 = start
                x1, y1 = end
                dx = x1 - x0
                dy = y1 - y0
                steps = max(abs(dx), abs(dy))
                for i in range(steps + 1):
                    xi = x0 + round(dx * i / steps)
                    yi = y0 + round(dy * i / steps)
                    white_cells.add((xi, yi))

        # 2. 색상 매핑
        color_grid = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                if self.map[y][x] == 1:
                    color_grid[y][x] = 0  # black
                elif (x, y) == agv_pos:
                    color_grid[y][x] = 1  # blue
                elif (x, y) == goal_pos or isinstance(self.map[y][x], str):
                    print(x, y)
                    color_grid[y][x] = 3  # red
                elif (x, y) in white_cells:
                    color_grid[y][x] = 4  # white (노드 또는 엣지 셀)
                else:
                    color_grid[y][x] = 2  # gray (나머지)

        cmap = ListedColormap(['black', 'blue', 'gray', 'red', 'white'])

        plt.figure(figsize=(10, 10))
        plt.imshow(color_grid, cmap=cmap)
        plt.grid(True, color='gray', linewidth=0.5)
        plt.xticks(np.arange(width))
        plt.yticks(np.arange(height))
        plt.title("Graph-based Grid (Nodes + Edges as White)")
        plt.show()



ctrl = Controller()
ctrl.graphing()
ctrl.visualize_grid_matplotlib()