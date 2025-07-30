import matplotlib.pyplot as plt
import numpy as np

class MatplotlibVisualizer:
    """
    Matplotlib을 사용하여 환경 상태를 시각화하고,
    스페이스바 입력을 기다리는 클래스.
    """
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.map_display = None
        self.is_paused = True
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        """키가 눌렸을 때 호출되는 이벤트 핸들러"""
        if event.key == ' ':
            self.is_paused = False # 스페이스바가 눌리면 대기 상태 해제

    def render_and_wait(self, env_instance):
        """
        현재 상태를 렌더링하고 스페이스바 입력을 기다립니다.
        """
        self.ax.clear()

        # 1. 맵 그리기
        map_data = env_instance.map
        self.ax.imshow(map_data, cmap='gray_r', origin='upper')

        # 3. AGV 위치, 목표 위치, 경로 그리기
        agv_list = env_instance.Get_AGV()
        for num, agv in agv_list.items():
            x, y = agv.pos
            gx, gy = agv.goal
            
            normalized_color = tuple(c / 255.0 for c in agv.color)
            
            # 현재 위치
            self.ax.scatter(x, y, color=normalized_color, s=100, label=f'AGV {num}', zorder=10)
            self.ax.text(x, y, str(num), color='white', ha='center', va='center', fontsize=8, fontweight='bold')

            # 목표 위치 (작은 점으로 표시)
            if agv.goal != (-1, -1): # 초기값(-1,-1)이 아닐 때만 표시
                self.ax.scatter(gx, gy, color=normalized_color, s=30, marker='x', zorder=9)
                # 현재 위치와 목표 위치를 선으로 연결
                self.ax.plot([x, gx], [y, gy], color=normalized_color, linestyle='--', linewidth=1, zorder=8)

        # 4. 제목 및 범례 설정 (안내 문구 추가)
        self.ax.set_title(f"Time: {env_instance.time} | Press SPACE to advance step")
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 스페이스바가 눌릴 때까지 대기
        self.is_paused = True
        while self.is_paused:
            plt.pause(0.1) # GUI 이벤트를 처리하며 대기

    def close(self):
        plt.ioff()
        plt.close(self.fig)