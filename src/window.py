import ctypes
from ctypes import wintypes
import time

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
