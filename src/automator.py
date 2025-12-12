import time
import numpy as np
import cv2
import pyautogui
from PIL import ImageGrab
import PIL.Image
import ctypes

# === Windows 底层键盘输入 (用于游戏) ===
SendInput = ctypes.windll.user32.SendInput

# C struct 定义
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# DirectInput 扫描码 (用于游戏)
# W 键的扫描码是 0x11, ESC 键的扫描码是 0x01
DIK_W = 0x11
DIK_ESCAPE = 0x01

KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

def press_key_direct(scan_code):
    """使用 DirectInput 扫描码按下按键"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, scan_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key_direct(scan_code):
    """使用 DirectInput 扫描码释放按键"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, scan_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def press_w_key():
    """模拟按下并释放 W 键 (游戏兼容)"""
    press_key_direct(DIK_W)
    time.sleep(0.05)  # 短暂按住
    release_key_direct(DIK_W)

def press_esc_key():
    """模拟按下并释放 ESC 键 (游戏兼容)"""
    press_key_direct(DIK_ESCAPE)
    time.sleep(0.05)  # 短暂按住
    release_key_direct(DIK_ESCAPE)

# Patch for older PIL versions
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from .config import CONFIG_MODES, GameConfig
from .window import WindowMgr
from .ocr import OCR
from .solver import Solver, TIME_LIMIT_PHASE1, TIME_LIMIT_PHASE2

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
        
        # 统计识别到的非零数字个数
        recognized_count = np.count_nonzero(self.grid)
        print(f"扫描完成，耗时: {time.time() - t_start:.2f}s，识别到 {recognized_count} 个数字")
        return recognized_count

    def validate_grid_solvability(self):
        """
        数论验证：检查棋盘是否可解。
        1. 检查所有数字的总和是否能被10整除
        2. 大数吞噬检查（贪心消耗模拟）
        
        Returns:
            tuple: (is_valid, error_message) - is_valid 为 True 表示验证通过
        """
        # 统计每个数字的出现次数 (1-9)
        count = {}
        for i in range(1, 10):
            count[i] = np.count_nonzero(self.grid == i)
        
        # === 检查 1：总和是否能被 10 整除 ===
        total_sum = sum(i * count[i] for i in range(1, 10))
        if total_sum % 10 != 0:
            return False, f"数论验证失败：数字总和 {total_sum} 无法被 10 整除"
        
        print(f"数论检查 1/2：总和 = {total_sum}，能被 10 整除 ✓")
        
        # === 检查 2：大数吞噬模拟 ===
        # 创建库存副本进行模拟
        stock = count.copy()
        
        # 处理 9：每个 9 必须消耗一个 1
        if stock[1] < stock[9]:
            return False, f"大数吞噬失败：9 需要 {stock[9]} 个 1，但只有 {stock[1]} 个"
        stock[1] -= stock[9]
        stock[9] = 0
        
        # 处理 8：每个 8 需要补 2 (优先用 2，否则用两个 1)
        need_8 = stock[8]
        # 优先消耗 2
        use_2 = min(stock[2], need_8)
        stock[2] -= use_2
        remaining_8 = need_8 - use_2
        # 剩余用两个 1
        need_1_for_8 = remaining_8 * 2
        if stock[1] < need_1_for_8:
            return False, f"大数吞噬失败：8 还需要 {need_1_for_8} 个 1，但只有 {stock[1]} 个"
        stock[1] -= need_1_for_8
        stock[8] = 0
        
        # 处理 7：每个 7 需要补 3 (优先 3，然后 2+1，最后 1+1+1)
        for _ in range(stock[7]):
            if stock[3] >= 1:
                stock[3] -= 1
            elif stock[2] >= 1 and stock[1] >= 1:
                stock[2] -= 1
                stock[1] -= 1
            elif stock[1] >= 3:
                stock[1] -= 3
            else:
                return False, f"大数吞噬失败：无法为 7 凑够补数 3"
        stock[7] = 0
        
        # 处理 6：每个 6 需要补 4 (优先 4，然后 3+1，2+2，2+1+1，1+1+1+1)
        for _ in range(stock[6]):
            if stock[4] >= 1:
                stock[4] -= 1
            elif stock[3] >= 1 and stock[1] >= 1:
                stock[3] -= 1
                stock[1] -= 1
            elif stock[2] >= 2:
                stock[2] -= 2
            elif stock[2] >= 1 and stock[1] >= 2:
                stock[2] -= 1
                stock[1] -= 2
            elif stock[1] >= 4:
                stock[1] -= 4
            else:
                return False, f"大数吞噬失败：无法为 6 凑够补数 4"
        stock[6] = 0
        
        print(f"数论检查 2/2：大数吞噬模拟通过 ✓")
        return True, ""
   
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
        
        is_first_run = True  # 用于区分初始化和重试
        
        while True:
            print("2. 执行初始点击操作...")
            cx1, cy1 = self.config.get_point(2118, 2000)
            pyautogui.click(cx1, cy1)
            
            # 初始化等待5秒，重试只等待1秒
            wait_time = 5 if is_first_run else 1
            print(f"3. 操作完成，等待 {wait_time} 秒准备识图...")
            time.sleep(wait_time)
            
            recognized_count = self.capture_and_recognize()
            
            # === OCR 验证：检查识别到的数字个数 ===
            expected_counts = [160, 135]  # 16x10 或 15x9 网格
            if recognized_count not in expected_counts:
                print(f"OCR 验证失败 ({recognized_count})，等待 1 秒后重试...")
                time.sleep(1)
                recognized_count = self.capture_and_recognize()
                
                if recognized_count not in expected_counts:
                    raise RuntimeError(
                        f"OCR 识别失败！识别到 {recognized_count} 个数字，"
                        f"但预期应为 {expected_counts} 之一。请检查游戏窗口是否正确显示。"
                    )
            print(f"OCR 验证通过：识别到 {recognized_count} 个数字")
            
            # === 数论验证：检查棋盘是否可解 ===
            is_valid, error_msg = self.validate_grid_solvability()
            if not is_valid:
                print(f"!!! {error_msg} !!!")
                print("正在执行恢复操作：按下 ESC 键...")
                press_esc_key()
                time.sleep(1)
                print("点击重试按钮 (2100, 1325)...")
                retry_x, retry_y = self.config.get_point(2100, 1325)
                pyautogui.click(retry_x, retry_y)
                print("等待 3 秒后重新开始...")
                time.sleep(3)
                is_first_run = False  # 重试不再是首次运行
                continue  # 重新开始循环
            print("数论验证通过 ✓")
            is_first_run = False  # 验证通过后，后续循环不再是首次运行
            
            # --- Simple Solver Benchmark ---
            print("=== 正在运行【基准算法】计算当前分数 ===")
            simple_score = self.solver.calculate_simple_score(self.grid)
            print(f">>> [基准贪心算法] 理论最高得分: {simple_score}")
            
            # --- God Brain V7 Integration ---
            # 第一阶段：使用约束模式，确保每一步后棋盘仍然理论可解
            print(f"=== 正在调用神之大脑进行演算 [约束模式] ({TIME_LIMIT_PHASE1}s) ===")
            path = self.solver.solve_hydra_constrained(self.grid)
            
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

            # === 第二轮操作：按下W键，重新识图并计算 ===
            print("=== 开始第二轮操作 ===")
            print("1. 模拟按下键盘 'W' 键...")
            press_w_key()
            
            print("2. 等待 2 秒...")
            time.sleep(2)
            
            print("3. 重新进行识图...")
            self.capture_and_recognize()
            
            # 第二阶段：使用普通模式（无约束），因为已经是最后一次机会
            print(f"4. 再次调用神之大脑进行演算 [普通模式] ({TIME_LIMIT_PHASE2}s)...")
            path_round2 = self.solver.solve_hydra(self.grid)
            
            if path_round2:
                print(f"=== 第二轮演算完成，准备执行 {len(path_round2)} 步操作 ===")
                for i, move in enumerate(path_round2):
                    r1, c1, r2, c2 = move
                    self.execute_move((r1, c1), (r2, c2))
                    self.update_internal_state((r1, c1), (r2, c2))
                    time.sleep(0.02)
                    
                    if i % 10 == 0:
                        print(f"进度: {i}/{len(path_round2)}")
                
                print(">>> 第二轮执行完毕 <<<")
            else:
                print("!!! 第二轮：神之大脑未找到解 !!!")
            
            print("=== 全部操作完成 ===")

            if not self.loop_mode:
                print("程序结束。")
                return

            print("等待 8 秒后重启循环...")
            time.sleep(8)
            is_first_run = True

