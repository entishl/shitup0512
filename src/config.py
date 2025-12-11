import numpy as np

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
