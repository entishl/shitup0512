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
GRID_ROWS = 16
GRID_COLS = 10
TARGET_SUM = 10

# 坐标参数 (基准分辨率 3840 * 2160)
START_X = 1455
START_Y = 488
STEP_X = 103.555
STEP_Y = 103.466

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
        # 计算结果是浮点数
        x = START_X + col * STEP_X
        y = START_Y + row * STEP_Y
        
        # 关键修改：先 round 四舍五入，再转 int
        return int(round(x)), int(round(y))

    def capture_and_recognize(self):
        """全量扫描：截取屏幕并识别整个棋盘"""
        print("正在进行全屏扫描和识别...")
        t_start = time.time()
        
        # 1. 计算截图区域 (注意转为 int)
        x1 = int(round(START_X - STEP_X / 2))
        y1 = int(round(START_Y - STEP_Y / 2))
        width = int(round(GRID_COLS * STEP_X))
        height = int(round(GRID_ROWS * STEP_Y))
        
        screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                # 2. 计算相对坐标 (关键修改：使用浮点运算后转整)
                # local_cx 是相对于截图左上角的中心点 X
                local_cx = int(round((c * STEP_X) + (STEP_X / 2)))
                local_cy = int(round((r * STEP_Y) + (STEP_Y / 2)))
                
                half_crop = CROP_SIZE // 2
                
                # 确保切片不越界（增加一点鲁棒性）
                y_min = max(0, local_cy - half_crop)
                y_max = min(img_bgr.shape[0], local_cy + half_crop)
                x_min = max(0, local_cx - half_crop)
                x_max = min(img_bgr.shape[1], local_cx + half_crop)

                cell_img = img_bgr[y_min:y_max, x_min:x_max]
                
                # ... 后续识别代码不变 ...
                if cell_img.size == 0: continue # 防止切片为空报错

                _, img_bytes = cv2.imencode('.png', cell_img)
                res = self.ocr.classification(img_bytes.tobytes())
                
                if res and res.isdigit():
                    self.grid[r][c] = int(res)
                else:
                    self.grid[r][c] = 0
        
        print(f"扫描完成，耗时: {time.time() - t_start:.2f}s")

    def find_first_move(self):
        """
        寻找可行解。
        注意：因为 points 是按行扫描生成的，所以返回的 start_pos 的行号一定 <= end_pos 的行号。
        也就是说，我们只会有 "向下" (含左下/右下) 或 "水平向右" 的拖拽，不会有向上拖拽。
        """
        # 1. 获取所有有效数字的坐标点
        points = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.grid[r][c] > 0:
                    points.append((r, c))
        
        # 2. 遍历点对
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                r1, c1 = points[i]
                r2, c2 = points[j]
                
                # 计算矩形边界
                r_start, r_end = min(r1, r2), max(r1, r2)
                c_start, c_end = min(c1, c2), max(c1, c2)
                
                # 提取子矩阵并求和
                sub_rect = self.grid[r_start:r_end+1, c_start:c_end+1]
                current_sum = np.sum(sub_rect)
                
                if current_sum == TARGET_SUM:
                    return ((r1, c1), (r2, c2))
                    
        return None

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
        
        move_duration = 0.1 + max_steps * 0.12 
        if move_duration <= 0:
            move_duration = 0.20

        # === 2. 动态计算偏移 (核心修正) ===
        # 偏移量 magnitude
        OFFSET = 10 
        
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
        
        print("2. 执行初始点击操作...")
        pyautogui.click(1877, 1685)
        time.sleep(0.1)
        pyautogui.click(2118, 2000)
        
        print("3. 操作完成，再次等待 3 秒准备识图...")
        time.sleep(3)
        
        self.capture_and_recognize()
        
        print("=== 开始自动消除循环 ===")
        consecutive_failures = 0
        
        while True:
            move = self.find_first_move()
            
            if move:
                consecutive_failures = 0
                self.execute_move(move[0], move[1])
                self.update_internal_state(move[0], move[1])
                time.sleep(0.15) 
            
            else:
                print("内存中无解，重新扫描屏幕校准...")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("连续多次扫描无解，程序退出。")
                    break
                    
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