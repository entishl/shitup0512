import numpy as np
from .config import GameConfig

class Solver:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def find_first_move(self, grid):
        """
        寻找可行解。
        注意：因为 points 是按行扫描生成的，所以返回的 start_pos 的行号一定 <= end_pos 的行号。
        也就是说，我们只会有 "向下" (含左下/右下) 或 "水平向右" 的拖拽，不会有向上拖拽。
        """
        # 1. 获取所有有效数字的坐标点
        points = []
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] > 0:
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
                sub_rect = grid[r_start:r_end+1, c_start:c_end+1]
                current_sum = np.sum(sub_rect)
                
                if current_sum == GameConfig.TARGET_SUM:
                    return ((r1, c1), (r2, c2))
                    
        return None

    def find_global_optimal_move(self, grid):
        """
        寻找全局最优解（全网格扫描版）：
        修正了旧版只扫描非零点的bug，现在可以识别以0为边缘或角落的矩形。
        """
        candidates = []
        
        # 1. 全网格遍历：尝试所有可能的矩形 (暴力扫描)
        # 只要矩形内的和为10，就是合法解，不管角落是不是0
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                # 为了减少重复计算，r2 从 r1 开始
                for r2 in range(r1, self.rows):
                    # c2 从 c1 开始，这样保证 (r1,c1) 永远是左上，(r2,c2) 永远是右下
                    for c2 in range(c1, self.cols):
                        
                        # 快速提取子矩阵求和
                        # 注意：切片是左闭右开，所以 end 要 +1
                        sub_rect = grid[r1:r2+1, c1:c2+1]
                        
                        # 只有和为10才是合法移动
                        if np.sum(sub_rect) == GameConfig.TARGET_SUM:
                            score = np.count_nonzero(sub_rect)
                            
                            # 过滤掉得分为0的纯0无效操作（虽然和是10不可能全0，但防一手）
                            if score == 0: continue

                            # 记录方案详情
                            candidates.append({
                                'p1': (r1, c1),
                                'p2': (r2, c2),
                                'r_range': (r1, r2),
                                'c_range': (c1, c2),
                                'score': score,
                                'area': (r2 - r1 + 1) * (c2 - c1 + 1),
                                'opportunity_cost': 0
                            })

        if not candidates:
            return None

        # 2. 计算冲突与机会成本
        # 这一步计算量大概是 N^2，当解特别多时可能会稍微花点时间，但在10x16网格下通常很快
        num_candidates = len(candidates)
        # 简单的碰撞检测缓存
        for i in range(num_candidates):
            cand_a = candidates[i]
            for j in range(i + 1, num_candidates):
                cand_b = candidates[j]
                
                # 判断矩形是否重叠
                overlap_r = max(cand_a['r_range'][0], cand_b['r_range'][0]) <= min(cand_a['r_range'][1], cand_b['r_range'][1])
                overlap_c = max(cand_a['c_range'][0], cand_b['c_range'][0]) <= min(cand_a['c_range'][1], cand_b['c_range'][1])
                
                if overlap_r and overlap_c:
                    # 发现冲突，互相增加机会成本
                    cand_a['opportunity_cost'] += cand_b['score']
                    cand_b['opportunity_cost'] += cand_a['score']

        # 3. 排序决策
        # 优先级: 得分高 > 破坏性小(机会成本低) > 面积大
        candidates.sort(key=lambda x: (x['score'], -x['opportunity_cost'], x['area']), reverse=True)
        
        best_move = candidates[0]
        
        # 调试用：如果解的角落是0，打印出来看看
        p1 = best_move['p1']
        p2 = best_move['p2']
        if grid[p1[0]][p1[1]] == 0 or grid[p2[0]][p2[1]] == 0:
             print(f"检测到含0边界的解: {p1}->{p2} (得分: {best_move['score']})")

        return (best_move['p1'], best_move['p2'])
