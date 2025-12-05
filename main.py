import cv2
import numpy as np
import pyautogui
import time
import PIL.Image

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import ddddocr
from PIL import ImageGrab

import ctypes
from ctypes import wintypes

# ================= 窗口管理 =================

class WindowMgr:
    def __init__(self):
        self._user32 = ctypes.windll.user32
        self._psapi = ctypes.windll.psapi
        self._kernel32 = ctypes.windll.kernel32
        # 设置 DPI 感知，确保在高 DPI 屏幕下获取真实的物理像素坐标
        try:
            self._user32.SetProcessDPIAware()
        except AttributeError:
            pass # 可能在非 Windows 或旧版 Windows 上

    def find_window(self, potential_titles, target_process_name=None):
        """根据标题和(可选)进程名寻找窗口句柄"""
        target_hwnd = None
        
        def callback(hwnd, _):
            nonlocal target_hwnd
            if not self._user32.IsWindowVisible(hwnd):
                return True
            
            length = self._user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
                
            buff = ctypes.create_unicode_buffer(length + 1)
            self._user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            
            if title in potential_titles:
                if target_process_name:
                    # 获取进程 ID
                    pid = ctypes.c_ulong()
                    self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    
                    # 尝试 1: 使用 GetModuleBaseName (需要较高权限)
                    # PROCESS_QUERY_INFORMATION (0x0400) | PROCESS_VM_READ (0x0010)
                    h_process = self._kernel32.OpenProcess(0x0410, False, pid)
                    
                    found_name = None
                    
                    if h_process:
                        exe_buff = ctypes.create_unicode_buffer(1024)
                        if self._psapi.GetModuleBaseNameW(h_process, None, exe_buff, 1024):
                            found_name = exe_buff.value
                        self._kernel32.CloseHandle(h_process)
                    
                    # 尝试 2: 如果失败，使用 GetProcessImageFileName (需要 PROCESS_QUERY_LIMITED_INFORMATION 0x1000)
                    if not found_name:
                        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                        h_process = self._kernel32.OpenProcess(0x1000, False, pid)
                        if h_process:
                            exe_path_buff = ctypes.create_unicode_buffer(1024)
                            if self._psapi.GetProcessImageFileNameW(h_process, exe_path_buff, 1024):
                                full_path = exe_path_buff.value
                                # 提取文件名
                                found_name = full_path.split('\\')[-1]
                            self._kernel32.CloseHandle(h_process)

                    if found_name:
                        print(f"找到窗口 '{title}' (PID: {pid.value})，进程名: {found_name}")
                        if found_name.lower() == target_process_name.lower():
                            target_hwnd = hwnd
                            return False # Found matched
                    else:
                        print(f"找到窗口 '{title}' (PID: {pid.value})，但无法读取进程名 (可能需要管理员权限)")
                
                else:
                    target_hwnd = hwnd
                    return False # Stop enumeration
            return True
            
        CMPFUNC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        self._user32.EnumWindows(CMPFUNC(callback), 0)
        return target_hwnd

    def get_window_rect(self, hwnd):
        """获取窗口客户区（内容区）的屏幕坐标和尺寸"""
        # 1. 检查是否最小化，如果是则还原
        if self._user32.IsIconic(hwnd):
            print("窗口处于最小化状态，正在还原...")
            # SW_RESTORE = 9
            self._user32.ShowWindow(hwnd, 9)
            time.sleep(0.5) # 等待动画
        
        # 2. 尝试置顶窗口 (可选，防止被遮挡)
        try:
            self._user32.SetForegroundWindow(hwnd)
            time.sleep(0.2)
        except Exception:
            pass

        rect = wintypes.RECT()
        self._user32.GetClientRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        
        # 将客户区左上角 (0,0) 转换为屏幕坐标
        pt = wintypes.POINT()
        pt.x = 0
        pt.y = 0
        self._user32.ClientToScreen(hwnd, ctypes.byref(pt))
        
        return pt.x, pt.y, w, h

# ================= 游戏配置与常量 =================

CONFIG_MODES = {
    "1": {
        "NAME": "10x16",
        "GRID_ROWS": 16,
        "GRID_COLS": 10,
        "REF_START_X": 1455,
        "REF_START_Y": 488,
        "REF_STEP_X": 103.555,
        "REF_STEP_Y": 103.466
    },
    "2": {
        "NAME": "9x15",
        "GRID_ROWS": 15,
        "GRID_COLS": 9,
        "REF_START_X": 1480,
        "REF_START_Y": 495,
        "REF_STEP_X": 110,
        "REF_STEP_Y": 110
    }
}

class GameConfig:
    # 游戏逻辑常量 (通用)
    TARGET_SUM = 10

    # 基准分辨率 (4k 全屏 3840x2160)
    REF_WIDTH = 3840
    REF_HEIGHT = 2160 # 仅作参考，主要用宽度计算缩放
    
    # 通用基准参数
    REF_CROP_SIZE = 50
    REF_OFFSET_MAGNITUDE = 10
    
    def __init__(self, mode_config, start_x_offset, start_y_offset, width, height):
        if width <= 0 or height <= 0:
            raise ValueError(f"窗口尺寸无效 ({width}x{height})，请检查窗口是否正常显示。")
            
        self.win_x = start_x_offset
        self.win_y = start_y_offset
        self.width = width
        self.height = height
        
        # 加载模式特定配置
        self.GRID_ROWS = mode_config["GRID_ROWS"]
        self.GRID_COLS = mode_config["GRID_COLS"]
        self.REF_START_X = mode_config["REF_START_X"]
        self.REF_START_Y = mode_config["REF_START_Y"]
        self.REF_STEP_X = mode_config["REF_STEP_X"]
        self.REF_STEP_Y = mode_config["REF_STEP_Y"]

        # 计算缩放比例 (以宽度为基准)
        # 假设游戏内容是固定的 16:9，如果窗口比例不同，可能会有黑边，
        # 这里默认窗口内容即为游戏内容（或者无黑边）
        self.scale = self.width / self.REF_WIDTH
        
        # 实时计算当前参数
        self.step_x = self.REF_STEP_X * self.scale
        self.step_y = self.REF_STEP_Y * self.scale
        self.crop_size = int(self.REF_CROP_SIZE * self.scale)
        self.offset_magnitude = self.REF_OFFSET_MAGNITUDE * self.scale
        
        # 起始点 (相对于屏幕绝对坐标)
        self.start_x_screen = self.win_x + (self.REF_START_X * self.scale)
        self.start_y_screen = self.win_y + (self.REF_START_Y * self.scale)

        print(f"[配置] 模式: {mode_config['NAME']} | 窗口尺寸: {self.width}x{self.height} | 缩放比: {self.scale:.4f} | 内容区原点: ({self.win_x}, {self.win_y})")

    def get_cell_center(self, row, col):
        x = self.start_x_screen + col * self.step_x
        y = self.start_y_screen + row * self.step_y
        return int(round(x)), int(round(y))

    def get_point(self, ref_x, ref_y):
        """将 4k 基准坐标转换为当前屏幕绝对坐标"""
        x = self.win_x + ref_x * self.scale
        y = self.win_y + ref_y * self.scale
        return int(round(x)), int(round(y))


# PyAutoGUI 设置
pyautogui.PAUSE = 0.005 
pyautogui.FAILSAFE = True

# ===========================================

class GameAutomator:
    def __init__(self, mode_key="1"):
        self.mode_config = CONFIG_MODES.get(mode_key, CONFIG_MODES["1"])
        self.rows = self.mode_config["GRID_ROWS"]
        self.cols = self.mode_config["GRID_COLS"]
        
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.ocr = ddddocr.DdddOcr()
        self.config = None
        self._init_window()
        print("初始化完成，OCR模型已加载。")

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
        for r in range(self.rows):
            for c in range(self.cols):
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
                
                if current_sum == GameConfig.TARGET_SUM:
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
        cx1, cy1 = self.config.get_point(2118, 2000)
        pyautogui.click(cx1, cy1)
        time.sleep(0.1)
        cx2, cy2 = self.config.get_point(1877, 1685)
        pyautogui.click(cx2, cy2)
        
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
    print("======= 注意 =======")
    print("请确保游戏在16:9下运行，本程序不支持其它比例")
    print("程序会产生大量拖拽动作，建议全屏模式下运行，或让游戏窗口尽量大，覆盖你的桌面文件")
    print("请右键选择管理员模式运行此程序，否则无法操作NIKKE")
    print("如果需要停止，将鼠标快速向左上角滑动")
    print("请先进入一次小游戏，暂停后【快速退出】，结算当次游戏后再执行程序")
    print("======= 注意 =======")
    print("请选择游戏模式:")
    print("1. 10x16 (默认)")
    print("2. 9x15")
    choice = input("请输入序号 (1/2): ").strip()
    
    if choice not in ["1", "2"]:
        print("输入无效，默认使用模式 1")
        choice = "1"
        
    bot = GameAutomator(choice)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n程序被用户强制停止。")