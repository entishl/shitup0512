import cv2
import numpy as np
import pyautogui
import time
import PIL.Image

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import ddddocr
from PIL import ImageGrab

# ================= 配置区域 =================

# 游戏参数
GRID_ROWS = 15
GRID_COLS = 9
TARGET_SUM = 10

# 坐标参数 (基准分辨率 3840 * 2160)
START_X = 1480
START_Y = 495
STEP_X = 110
STEP_Y = 110

# 识别区域大小
CROP_SIZE = 50 

# PyAutoGUI 设置
pyautogui.PAUSE = 0.005 
pyautogui.FAILSAFE = True

# ===========================================

class GameAutomator:
    def __init__(self):
        self.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        self.ocr = ddddocr.DdddOcr()
        print("初始化完成，OCR模型已加载。")

    def get_cell_center(self, row, col):
        """计算第 row 行, col 列的屏幕绝对坐标"""
        x = START_X + col * STEP_X
        y = START_Y + row * STEP_Y
        return int(x), int(y)

    def capture_and_recognize(self):
        """全量扫描：截取屏幕并识别整个棋盘"""
        print("正在进行全屏扫描和识别...")
        t_start = time.time()
        
        x1 = START_X - STEP_X // 2
        y1 = START_Y - STEP_Y // 2
        width = GRID_COLS * STEP_X
        height = GRID_ROWS * STEP_Y
        
        screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                local_cx = (c * STEP_X) + (STEP_X // 2)
                local_cy = (r * STEP_Y) + (STEP_Y // 2)
                
                half_crop = CROP_SIZE // 2
                cell_img = img_bgr[local_cy-half_crop:local_cy+half_crop, 
                                   local_cx-half_crop:local_cx+half_crop]
                
                _, img_bytes = cv2.imencode('.png', cell_img)
                res = self.ocr.classification(img_bytes.tobytes())
                
                if res and res.isdigit():
                    self.grid[r][c] = int(res)
                else:
                    self.grid[r][c] = 0
        
        print(f"扫描完成，耗时: {time.time() - t_start:.2f}s")

    def find_first_move(self):
        """在内存中寻找可行解"""
        for r1 in range(GRID_ROWS):
            for c1 in range(GRID_COLS):
                if self.grid[r1][c1] == 0:
                    continue

                for r2 in range(r1, GRID_ROWS):
                    start_c2 = c1 if r2 > r1 else c1 
                    for c2 in range(start_c2, GRID_COLS):
                        sub_rect = self.grid[r1:r2+1, c1:c2+1]
                        current_sum = np.sum(sub_rect)
                        
                        if current_sum == TARGET_SUM:
                            return ((r1, c1), (r2, c2))
                        elif current_sum > TARGET_SUM:
                            break 
        return None

    def execute_move(self, start_pos, end_pos):
        """执行鼠标拖拽，根据距离动态计算耗时"""
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        x1, y1 = self.get_cell_center(r1, c1)
        x2, y2 = self.get_cell_center(r2, c2)
        
        # === 动态计算 duration ===
        dist_r = abs(r2 - r1)
        dist_c = abs(c2 - c1)
        max_steps = max(dist_r, dist_c)
        
        move_duration = 0.1 + max_steps * 0.15
        if move_duration <= 0:
            move_duration = 0.25

        print(f"执行消除: ({r1},{c1}) -> ({r2},{c2}) | 跨度: {max_steps}格 | 耗时: {move_duration:.2f}s")
        
        pyautogui.moveTo(x1, y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2 + 10, y2 + 10, duration=move_duration) 
        pyautogui.mouseUp()

    def update_internal_state(self, start_pos, end_pos):
        """直接在内存中更新棋盘状态"""
        r1, c1 = start_pos
        r2, c2 = end_pos
        self.grid[r1:r2+1, c1:c2+1] = 0

    def run(self):
        print("=== 程序启动流程 ===")
        print("1. 等待 3 秒...")
        time.sleep(3)
        
        print("2. 执行初始点击操作...")
        pyautogui.click(1877, 1685)  # 第一次点击
        time.sleep(0.1)              # 间隔 0.1s
        pyautogui.click(2118, 2000)  # 第二次点击
        
        print("3. 操作完成，再次等待 3 秒准备识图...")
        time.sleep(3)
        
        # 首次全屏扫描
        self.capture_and_recognize()
        
        print("=== 开始自动消除循环 ===")
        while True:
            # 1. 内存计算
            move = self.find_first_move()
            
            if move:
                # 2. 执行操作 (含动态延时)
                self.execute_move(move[0], move[1])
                # 3. 更新内存
                self.update_internal_state(move[0], move[1])
                # 4. 短暂等待消除动画
                time.sleep(0.1) 
            
            else:
                print("内存中无解，重新扫描屏幕校准...")
                self.capture_and_recognize()
                
                if self.find_first_move():
                    print("发现新解，继续运行。")
                    continue
                else:
                    print("确认无解，程序结束。")
                    break

if __name__ == "__main__":
    bot = GameAutomator()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n程序被用户强制停止。")