from Controller import controller
from Environment import ENV
import Funct
import matplotlib.pyplot as plt

env = ENV()

grid = env.controller.grid

import matplotlib.pyplot as plt
import numpy as np

def visualize_grid(grid):
    height, width = grid.shape
    fig, ax = plt.subplots(figsize=(width / 3, height / 3))
    
    cax = ax.matshow(grid, cmap='binary_r', origin='upper')
    
    # 그리드 경계선
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    
    # 눈금 제거
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    plt.title("Grid (0: White, 1: Black)")
    plt.show()

visualize_grid(grid)