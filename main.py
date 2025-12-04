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
        self.ocr = ddddocr.DdddOcr() # 关闭广告打印

    def get_cell_center(self, row, col):
        """计算第 row 行, col 列的屏幕绝对坐标"""
        x = START_X + col * STEP_X
        y = START_Y + row * STEP_Y
        return int(x), int(y)

    def capture_and_recognize(self):
        """截取屏幕并识别整个棋盘，更新 self.grid"""
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
        
    def find_first_move(self):
        """
        寻找矩形区域和为 TARGET_SUM (10) 的移动方案。
        """
        # 遍历矩形左上角
        for r1 in range(GRID_ROWS):
            for c1 in range(GRID_COLS):
                
                # 假设动作必须从非空数字开始（大部分游戏逻辑如此）
                if self.grid[r1][c1] == 0:
                    continue

                # 遍历矩形右下角
                for r2 in range(r1, GRID_ROWS):
                    # c2 的起始位置：
                    # 如果在同一行(r1==r2)，c2必须从 c1+1 开始（如果是两个数）或者 c1 开始（如果是单数10）
                    # 为了通用性，这里从 c1 开始
                    for c2 in range(c1, GRID_COLS):
                        
                        # 获取矩形切片
                        sub_rect = self.grid[r1:r2+1, c1:c2+1]
                        current_sum = np.sum(sub_rect)
                        
                        if current_sum == TARGET_SUM:
                            return ((r1, c1), (r2, c2))
                        elif current_sum > TARGET_SUM:
                            # 当前行往右加已经爆了，停止内层循环，换下一行继续尝试
                            break
        return None

    def execute_move(self, start_pos, end_pos):
        """执行鼠标拖拽"""
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        x1, y1 = self.get_cell_center(r1, c1)
        x2, y2 = self.get_cell_center(r2, c2)
        
        print(f"执行消除: ({r1},{c1}) -> ({r2},{c2})")
        
        pyautogui.moveTo(x1, y1)
        pyautogui.mouseDown()
        # 稍微增加一点移动时间，确保游戏能识别到拖拽轨迹
        pyautogui.moveTo(x2, y2, duration=0.30) 
        pyautogui.mouseUp()

    def run(self):
        print("程序将在3秒后开始，请切换到游戏界面...")
        time.sleep(3)
        
        while True:
            t0 = time.time()
            self.capture_and_recognize()
            # print(f"识别耗时: {time.time()-t0:.2f}s")
            
            # 打印一下当前的 grid (可选，用于调试)
            # print(self.grid)

            move = self.find_first_move()
            
            if move:
                self.execute_move(move[0], move[1])
                # 等待消除动画，防止识别到消除中的状态
                time.sleep(0.1) 
            else:
                print("未找到可消除的数字组合，程序结束。")
                break

if __name__ == "__main__":
    bot = GameAutomator()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("程序被用户强制停止。")