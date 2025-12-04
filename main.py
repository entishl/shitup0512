import cv2
import numpy as np
import pyautogui
import time
import PIL.Image # 引入 PIL

# ================= 修复 Pillow 10.0+ 报错的问题 =================
# ddddocr 依赖旧版 Pillow 的 ANTIALIAS，新版移除了，这里手动补上
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# ==============================================================

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

# 识别区域大小 (只截取数字中心)
CROP_SIZE = 50 

# PyAutoGUI 设置
pyautogui.PAUSE = 0.01 
pyautogui.FAILSAFE = True

# ===========================================

class GameAutomator:
    def __init__(self):
        self.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        
        # 初始化 ddddocr
        self.ocr = ddddocr.DdddOcr()

    def get_cell_center(self, row, col):
        """计算第 row 行, col 列的屏幕绝对坐标"""
        x = START_X + col * STEP_X
        y = START_Y + row * STEP_Y
        return int(x), int(y)

    def capture_and_recognize(self):
        """截取屏幕并识别整个棋盘，更新 self.grid"""
        # 计算覆盖区域
        x1 = START_X - STEP_X // 2
        y1 = START_Y - STEP_Y // 2
        width = GRID_COLS * STEP_X
        height = GRID_ROWS * STEP_Y
        
        # 截图
        screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 遍历每个格子
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                # 截取小图
                local_cx = (c * STEP_X) + (STEP_X // 2)
                local_cy = (r * STEP_Y) + (STEP_Y // 2)
                
                half_crop = CROP_SIZE // 2
                cell_img = img_bgr[local_cy-half_crop:local_cy+half_crop, 
                                   local_cx-half_crop:local_cx+half_crop]
                
                # 识别
                _, img_bytes = cv2.imencode('.png', cell_img)
                res = self.ocr.classification(img_bytes.tobytes())
                
                if res and res.isdigit():
                    self.grid[r][c] = int(res)
                else:
                    self.grid[r][c] = 0
        
    def find_first_move(self):
        """
        只寻找【第一个】可行的移动方案。
        """
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.grid[r][c] == 0:
                    continue
                
                # --- 1. 横向尝试 (向右) ---
                current_sum = self.grid[r][c]
                
                for k in range(c + 1, GRID_COLS):
                    val = self.grid[r][k]
                    if val == 0: continue # 跳过空白
                    
                    current_sum += val
                    
                    if current_sum == TARGET_SUM:
                        return ((r, c), (r, k))
                    elif current_sum > TARGET_SUM:
                        break 
                
                # --- 2. 纵向尝试 (向下) ---
                current_sum = self.grid[r][c]
                
                for k in range(r + 1, GRID_ROWS):
                    val = self.grid[k][c]
                    if val == 0: continue
                    
                    current_sum += val
                    
                    if current_sum == TARGET_SUM:
                        return ((r, c), (k, c))
                    elif current_sum > TARGET_SUM:
                        break

        return None

    def execute_move(self, start_pos, end_pos):
        """执行鼠标拖拽"""
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        x1, y1 = self.get_cell_center(r1, c1)
        x2, y2 = self.get_cell_center(r2, c2)
        
        print(f"执行消除: 行{r1}列{c1} ({self.grid[r1][c1]}) -> 行{r2}列{c2} ({self.grid[r2][c2]})")
        
        pyautogui.moveTo(x1, y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0.2) 
        pyautogui.mouseUp()

    def run(self):
        print("程序将在3秒后开始，请切换到游戏界面...")
        time.sleep(3)
        
        while True:
            # 1. 扫描当前屏幕
            self.capture_and_recognize()
            
            # 2. 寻找一个解
            move = self.find_first_move()
            
            if move:
                # 3. 执行
                self.execute_move(move[0], move[1])
                # 4. 等待动画
                time.sleep(0.3)
            else:
                print("未找到可消除的数字组合，程序结束。")
                break

if __name__ == "__main__":
    bot = GameAutomator()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("程序被用户强制停止。")