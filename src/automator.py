import time
import numpy as np
import cv2
import pyautogui
from PIL import ImageGrab
import PIL.Image

# Patch for older PIL versions
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from .config import CONFIG_MODES, GameConfig
from .window import WindowMgr
from .ocr import OCR
from .solver import Solver, TIME_LIMIT_SECONDS

# PyAutoGUI 设置
pyautogui.PAUSE = 0.005 
pyautogui.FAILSAFE = True

class GameAutomator:
    def __init__(self, mode_key="1", loop_mode=False):
        self.mode_config = CONFIG_MODES.get(mode_key, CONFIG_MODES["1"])
        self.loop_mode = loop_mode
        self.rows = self.mode_config["GRID_ROWS"]
        self.cols = self.mode_config["GRID_COLS"]
        
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.config = None
        
        self.ocr = OCR()
        self.solver = Solver(self.rows, self.cols)
        
        self._init_window()
        print("初始化完成。")

    def _init_window(self):
        win_mgr = WindowMgr()
        titles = ["NIKKE", "勝利女神：妮姬", "胜利女神：新的希望"]
        # 精确查找 nikke.exe 进程的窗口，防止误判启动器
        hwnd = win_mgr.find_window(titles, target_process_name="nikke.exe")
        
        if not hwnd:
            raise Exception(f"未找到游戏窗口! 请确保游戏已启动，窗口标题在以下列表中: {titles}")
            
        x, y, w, h = win_mgr.get_window_rect(hwnd)
        self.config = GameConfig(self.mode_config, x, y, w, h)

    def get_cell_center(self, row, col):
        """计算第 row 行, col 列的屏幕绝对坐标"""
        return self.config.get_cell_center(row, col)

    def capture_and_recognize(self):
        """全量扫描：截取屏幕并识别整个棋盘"""
        print("正在进行全屏扫描和识别...")
        t_start = time.time()
        
        # 1. 计算截图区域
        # 基于配置计算覆盖整个网格的区域
        # 左上角基准 (0,0) 的中心点
        c00_x, c00_y = self.config.get_cell_center(0, 0)
        
        # 为了稳健，我们可以截取包含所有格子的矩形
        # 左边 = 第0列中心 - 半个步长
        # 上边 = 第0行中心 - 半个步长
        x1 = int(round(c00_x - self.config.step_x / 2))
        y1 = int(round(c00_y - self.config.step_y / 2))
        
        # 宽度 = 列数 * 步长
        width = int(round(self.cols * self.config.step_x))
        height = int(round(self.rows * self.config.step_y))
        
        screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        for r in range(self.rows):
            for c in range(self.cols):
                # 2. 计算相对坐标
                # 相对于截图区域左上角的坐标
                # 绝对中心
                abs_cx, abs_cy = self.config.get_cell_center(r, c)
                
                # 相对中心 = 绝对中心 - 截图左上角
                local_cx = abs_cx - x1
                local_cy = abs_cy - y1
                
                half_crop = self.config.crop_size // 2
                
                # 确保切片不越界
                y_min = max(0, local_cy - half_crop)
                y_max = min(img_bgr.shape[0], local_cy + half_crop)
                x_min = max(0, local_cx - half_crop)
                x_max = min(img_bgr.shape[1], local_cx + half_crop)

                cell_img = img_bgr[y_min:y_max, x_min:x_max]
                
                res = self.ocr.recognize(cell_img)
                
                if res is not None:
                    self.grid[r][c] = res
                else:
                    self.grid[r][c] = 0
        
        print(f"扫描完成，耗时: {time.time() - t_start:.2f}s")
   
    def execute_move(self, start_pos, end_pos):
        """执行鼠标拖拽，根据移动向量动态计算偏移方向"""
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        x1, y1 = self.get_cell_center(r1, c1)
        x2, y2 = self.get_cell_center(r2, c2)
        
        # === 1. 动态计算 duration ===
        dist_r = abs(r2 - r1)
        dist_c = abs(c2 - c1)
        max_steps = max(dist_r, dist_c)
        
        move_duration = 0.22

        # === 2. 动态计算偏移 (核心修正) ===
        # 偏移量 magnitude
        OFFSET = self.config.offset_magnitude
        
        # X轴偏移逻辑：
        # 如果终点在起点右边 (c2 >= c1)，向右偏移 (+10)
        # 如果终点在起点左边 (c2 < c1)，向左偏移 (-10) -> 解决 "变短" 问题
        offset_x = OFFSET if c2 >= c1 else -OFFSET
        
        # Y轴偏移逻辑：
        # 如果终点在起点下方 (r2 >= r1)，向下偏移 (+10)
        # 如果终点在起点上方 (r2 < r1)，向上偏移 (-10)
        offset_y = OFFSET if r2 >= r1 else -OFFSET

        target_x = x2 + offset_x
        target_y = y2 + offset_y

        # 打印调试信息
        dir_str = []
        if r2 > r1: dir_str.append("下")
        elif r2 < r1: dir_str.append("上")
        if c2 > c1: dir_str.append("右")
        elif c2 < c1: dir_str.append("左")
        direction = "".join(dir_str) if dir_str else "原点"
        
        print(f"执行消除: ({r1},{c1})->({r2},{c2}) | 方向:{direction} | 目标偏移:({offset_x:+},{offset_y:+}) | 耗时:{move_duration:.2f}s")
        
        pyautogui.moveTo(x1, y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(target_x, target_y, duration=move_duration)
        time.sleep(0.1)  # 在终点停滞0.1秒后再松开鼠标
        pyautogui.mouseUp()

    def update_internal_state(self, start_pos, end_pos):
        """更新内存状态"""
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        r_start, r_end = min(r1, r2), max(r1, r2)
        c_start, c_end = min(c1, c2), max(c1, c2)
        
        self.grid[r_start:r_end+1, c_start:c_end+1] = 0

    def run(self):
        print("=== 程序启动流程 ===")
        print("1. 等待 3 秒...")
        time.sleep(3)
        
        while True:
            print("2. 执行初始点击操作...")
            cx1, cy1 = self.config.get_point(2118, 2000)
            pyautogui.click(cx1, cy1)
            
            print("3. 操作完成，再次等待 5 秒准备识图...")
            time.sleep(5)
            
            self.capture_and_recognize()
            
            # --- Simple Solver Benchmark ---
            print("=== 正在运行【基准算法】计算当前分数 ===")
            simple_score = self.solver.calculate_simple_score(self.grid)
            print(f">>> [基准贪心算法] 理论最高得分: {simple_score}")
            
            # --- God Brain V7 Integration ---
            print(f"=== 正在调用神之大脑进行演算 ({TIME_LIMIT_SECONDS}s) ===")
            path = self.solver.solve_hydra(self.grid)
            
            if path:
                print(f"=== 演算完成，准备执行 {len(path)} 步操作 ===")
                for i, move in enumerate(path):
                    r1, c1, r2, c2 = move
                    # execute_move expects start_pos, end_pos tuples
                    self.execute_move((r1, c1), (r2, c2))
                    
                    # Update internal state (optional but good for visuals if we had a UI)
                    self.update_internal_state((r1, c1), (r2, c2))
                    
                    # Small delay between moves to prevent input flooding
                    time.sleep(0.02)
                    
                    if i % 10 == 0:
                        print(f"进度: {i}/{len(path)}")
                
                print(">>> 本轮执行完毕 <<<")
            else:
                print("!!! 神之大脑未找到解 !!!")

            if not self.loop_mode:
                print("程序结束。")
                return

            print("等待 5 秒后重启循环...")
            time.sleep(5)

