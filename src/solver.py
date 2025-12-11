import numpy as np
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import GameConfig

# =============================================================================
# 配置参数 - 可在此处修改
# =============================================================================
TIME_LIMIT_SECONDS = 30  # 求解器时间限制（秒）

# --- Numba Check ---
try:
    from numba import njit, int8, int32, float32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print(">> [Warning] Numba not found! Solver will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# --- Core V7 Kernel Functions (Top-level for pickling) ---

@njit(fastmath=True, nogil=True, cache=True)
def _calc_prefix_sum(vals, rows, cols):
    P = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for r in range(rows):
        row_sum = 0
        for c in range(cols):
            row_sum += vals[r * cols + c]
            P[r + 1][c + 1] = P[r][c + 1] + row_sum
    return P

@njit(fastmath=True, nogil=True)
def _get_rect_sum(P, r1, c1, r2, c2):
    return P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]

@njit(fastmath=True, nogil=True)
def _get_rect_count(P_count, r1, c1, r2, c2):
    return P_count[r2+1][c2+1] - P_count[r1][c2+1] - P_count[r2+1][c1] + P_count[r1][c1]

@njit(fastmath=True, nogil=True)
def _count_islands(map_data, rows, cols):
    islands = 0
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if map_data[idx] == 1:
                if r > 0 and map_data[(r-1)*cols + c] == 1: continue
                if r < rows - 1 and map_data[(r+1)*cols + c] == 1: continue
                if c > 0 and map_data[r*cols + (c-1)] == 1: continue
                if c < cols - 1 and map_data[r*cols + (c+1)] == 1: continue
                islands += 1
    return islands

@njit(fastmath=True, nogil=True, cache=True)
def _build_active_indices(map_data, size):
    """
    Build array of active cell indices using simple loop.
    Faster than np.where() due to avoiding Python-NumPy overhead.
    """
    # First pass: count active cells
    count = 0
    for i in range(size):
        if map_data[i] == 1:
            count += 1
    
    # Second pass: fill indices array
    result = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(size):
        if map_data[i] == 1:
            result[idx] = i
            idx += 1
    return result

@njit(fastmath=True, nogil=True)
def _evaluate_state(score, map_data, rows, cols, w_island, w_fragment, remaining_count=-1):
    """
    Evaluate state heuristic score.
    Args:
        remaining_count: Pre-computed count of remaining active cells.
                        If -1, will be calculated from map_data (slower).
    """
    # 1. Basic Aggression
    h = float(score * 2000)
    
    # 2. Remaining Count (use pre-computed if available)
    if remaining_count < 0:
        remaining_count = 0
        for i in range(rows * cols):
            if map_data[i] == 1: remaining_count += 1
    
    # 3. Island Penalty (Dynamic)
    if w_island > 0:
        islands = _count_islands(map_data, rows, cols)
        
        # [V7 Feature] Endgame Panic
        # < 20 items left, island penalty x5.0, forcing clearance
        panic_multiplier = 1.0
        if remaining_count < 20 and remaining_count > 0:
            panic_multiplier = 5.0
            
        h -= islands * w_island * panic_multiplier
        
        if remaining_count < 10:
            h -= remaining_count * w_island * 2.0
    
    # 4. Center Gravity
    if w_fragment > 0 and remaining_count > 30:
        center_mass = 0
        center_r, center_c = rows // 2, cols // 2
        for r in range(rows):
            for c in range(cols):
                if map_data[r * cols + c] == 1:
                    dist = abs(r - center_r) + abs(c - center_c)
                    center_mass += (20 - dist)
        h -= center_mass * w_fragment
    
    # 5. Random Noise (S/L Optimization)
    noise_level = 50.0
    if w_island < 20 and w_fragment < 1:
        noise_level = 2000.0 # Gamble early
        
    if remaining_count < 30:
        noise_level *= 0.2 # Stable late game
        
    h += np.random.random() * noise_level
    return h

@njit(fastmath=True, nogil=True)
def _fast_scan_rects_v6(map_data, vals, rows, cols, active_indices):
    """
    Scan for valid rectangles and return moves along with computed prefix sums.
    Returns: (moves, P_val, P_cnt) - reuse P_val/P_cnt to avoid recalculation.
    """
    moves = []
    n_active = len(active_indices)
    current_vals = np.zeros(rows * cols, dtype=np.int32)
    current_counts = np.zeros(rows * cols, dtype=np.int32)
    for i in range(rows * cols):
        if map_data[i] == 1:
            current_vals[i] = vals[i]
            current_counts[i] = 1
    P_val = _calc_prefix_sum(current_vals, rows, cols)
    P_cnt = _calc_prefix_sum(current_counts, rows, cols)
    for i in range(n_active):
        for j in range(i, n_active):
            idx1 = active_indices[i]; idx2 = active_indices[j]
            
            # Early pruning: if corner values already exceed 10, skip
            # This is cheaper than prefix sum query and prunes ~44% of pairs
            if vals[idx1] + vals[idx2] > 10:
                continue
            
            r1_raw = idx1 // cols; c1_raw = idx1 % cols
            r2_raw = idx2 // cols; c2_raw = idx2 % cols
            min_r = min(r1_raw, r2_raw); max_r = max(r1_raw, r2_raw)
            min_c = min(c1_raw, c2_raw); max_c = max(c1_raw, c2_raw)
            if _get_rect_sum(P_val, min_r, min_c, max_r, max_c) != 10: continue
            count = _get_rect_count(P_cnt, min_r, min_c, max_r, max_c)
            moves.append((min_r, min_c, max_r, max_c, count))
    return moves, P_val, P_cnt

@njit(fastmath=True, nogil=True)
def _apply_move_fast(map_data, rect, cols):
    new_map = map_data.copy()
    r1, c1, r2, c2 = rect
    for r in range(r1, r2 + 1):
        base = r * cols
        for c in range(c1, c2 + 1):
            new_map[base + c] = 0
    return new_map

@njit(fastmath=True, nogil=True)
def _solve_greedy_numba(map_data, vals, rows, cols):
    """
    Numba-accelerated Greedy Solver.
    Returns total score.
    """
    score = 0
    current_map = map_data.copy()
    
    # Pre-allocate arrays to avoid repeated allocation in loop if possible, 
    # but inside Numba allocating small arrays is okay.
    
    while True:
        # 1. Identify active cells
        active_indices = []
        for i in range(rows * cols):
            if current_map[i] == 1:
                active_indices.append(i)
        
        n_active = len(active_indices)
        if n_active < 2:
            break
            
        # 2. Build Prefix Sums
        d_vals = np.zeros(rows * cols, dtype=np.int32)
        d_cnts = np.zeros(rows * cols, dtype=np.int32)
        
        # Fill only active values
        for idx in active_indices:
            d_vals[idx] = vals[idx]
            d_cnts[idx] = 1
            
        P_val = _calc_prefix_sum(d_vals, rows, cols)
        P_cnt = _calc_prefix_sum(d_cnts, rows, cols)
        
        # 3. Scan for First Move
        found_move = False
        r1_best, c1_best, r2_best, c2_best = -1, -1, -1, -1
        move_gain = 0
        
        for i in range(n_active):
            for j in range(i + 1, n_active):
                idx1 = active_indices[i]
                idx2 = active_indices[j]
                
                r1 = idx1 // cols; c1 = idx1 % cols
                r2 = idx2 // cols; c2 = idx2 % cols
                
                min_r = min(r1, r2); max_r = max(r1, r2)
                min_c = min(c1, c2); max_c = max(c1, c2)
                
                s = _get_rect_sum(P_val, min_r, min_c, max_r, max_c)
                
                if s == 10:
                    # Found first move
                    r1_best, c1_best, r2_best, c2_best = min_r, min_c, max_r, max_c
                    move_gain = _get_rect_count(P_cnt, min_r, min_c, max_r, max_c)
                    found_move = True
                    break
            if found_move: break
            
        if found_move:
            # Apply Move
            score += move_gain
            # Inline apply for speed
            for r in range(r1_best, r2_best + 1):
                base = r * cols
                for c in range(c1_best, c2_best + 1):
                    current_map[base + c] = 0
        else:
            break
            
    return score

def _unroll_path_chain(path_chain):
    """
    Unroll a tuple-chain path into a list.
    Path chain format: ((r1,c1,r2,c2), parent_chain) or None for empty.
    Returns: list of [r1, c1, r2, c2] moves in order.
    """
    result = []
    current = path_chain
    while current is not None:
        move, current = current
        result.append(list(move))
    result.reverse()  # Reverse since we built it backwards
    return result

def _run_core_search_logic(start_map, vals_arr, rows, cols, beam_width, search_mode, start_score, start_path, weights, max_depth=160):
    w_island = weights.get('w_island', 0)
    w_fragment = weights.get('w_fragment', 0)
    
    # Calculate initial remaining count
    initial_remaining = 0
    for i in range(rows * cols):
        if start_map[i] == 1: initial_remaining += 1
    
    # Convert initial path list to tuple chain format
    # Path chain: ((move_tuple), parent_chain) or None
    initial_path_chain = None
    for move in start_path:
        initial_path_chain = (tuple(move), initial_path_chain)
    
    initial_h = _evaluate_state(start_score, start_map, rows, cols, w_island, w_fragment, initial_remaining)
    current_beam = [{'map': start_map, 'path': initial_path_chain, 'score': start_score, 'h_score': initial_h, 'remaining': initial_remaining}]
    best_state_in_run = current_beam[0]
    
    for _ in range(max_depth):
        # [V7 Feature] Dynamic Beam Funnel
        current_max_score = best_state_in_run['score']
        effective_beam_width = beam_width
        
        # Endgame burst
        if current_max_score > 120:
            effective_beam_width = int(beam_width * 3.0)
        elif current_max_score > 80:
            effective_beam_width = int(beam_width * 1.5)
            
        next_candidates = []
        found_any_move = False
        
        for state in current_beam:
            # Use Numba-jitted function instead of np.where (faster)
            active_indices = _build_active_indices(state['map'], rows * cols)
            if len(active_indices) < 2:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue

            # Get moves and prefix sums
            raw_moves, _P_val, P_cnt = _fast_scan_rects_v6(state['map'], vals_arr, rows, cols, active_indices)
            # Get parent's remaining count from prefix sum (O(1) lookup)
            parent_remaining = P_cnt[rows][cols]
            
            if not raw_moves:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue
            
            valid_moves_for_state = []
            for m in raw_moves:
                count = m[4]
                rule_pass = False
                if search_mode == 'classic':
                    if count == 2: rule_pass = True
                else: 
                    if count >= 2: rule_pass = True
                if rule_pass: valid_moves_for_state.append(m)
            
            if not valid_moves_for_state:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue

            found_any_move = True
            
            # Sort + Truncate
            valid_moves_for_state.sort(key=lambda x: x[4], reverse=True)
            window_size = 60 if current_max_score < 120 else 100
            top_moves = valid_moves_for_state[:window_size]
            
            for move in top_moves:
                r1, c1, r2, c2, count = move
                new_map = _apply_move_fast(state['map'], (r1, c1, r2, c2), cols)
                new_score = state['score'] + count
                # O(1) remaining count: parent_remaining - cleared_count
                new_remaining = parent_remaining - count
                h = _evaluate_state(new_score, new_map, rows, cols, w_island, w_fragment, new_remaining)
                # O(1) path extension using tuple chain (cons-list pattern)
                new_path_chain = ((r1, c1, r2, c2), state['path'])
                next_candidates.append({'map': new_map, 'path': new_path_chain, 'score': new_score, 'h_score': h, 'remaining': new_remaining})

        if not found_any_move or not next_candidates: break
        
        next_candidates.sort(key=lambda x: x['h_score'], reverse=True)
        
        # [Optimization] State Deduplication
        # Remove duplicate map configurations to keep the beam diverse.
        # Since we sorted by score/h_score, we keep the best path to any given state.
        unique_beam = []
        seen_states = set()
        
        for cand in next_candidates:
            # Hash the map state. 
            # Using bytes is fast enough for 10x16 grids.
            state_hash = cand['map'].tobytes()
            if state_hash in seen_states:
                continue
            seen_states.add(state_hash)
            unique_beam.append(cand)
            if len(unique_beam) >= effective_beam_width:
                break
                
        current_beam = unique_beam
        
        if current_beam and current_beam[0]['score'] > best_state_in_run['score']:
            best_state_in_run = current_beam[0]
    
    # Unroll path chain to list before returning
    best_state_in_run['path'] = _unroll_path_chain(best_state_in_run['path'])
    return best_state_in_run

def _solve_process_hydra(args):
    map_list, val_list, rows, cols, beam_width, mode, seed, time_limit, personality = args
    safe_seed = seed % (2**32 - 1)
    np.random.seed(safe_seed)
    random.seed(safe_seed)
    
    initial_map_arr = np.array(map_list, dtype=np.int8)
    vals_arr = np.array(val_list, dtype=np.int8)
    
    weights = {'w_island': personality.get('w_island', 0), 'w_fragment': personality.get('w_fragment', 0)}
    start_time = time.time()
    
    # 1. Base Run
    base_state = None
    if mode == 'god':
        p1_weights = weights.copy()
        if p1_weights['w_island'] > 0: p1_weights['w_island'] *= 0.5 
        p1 = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, 'classic', 0, [], p1_weights)
        p2 = _run_core_search_logic(p1['map'], vals_arr, rows, cols, beam_width, 'omni', p1['score'], p1['path'], weights)
        base_state = p2
    else:
        base_state = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, mode, 0, [], weights)
        
    best_final_state = base_state
    
    # 2. Time Traveler Loop (S/L Loop)
    iteration = 0
    max_score_history = best_final_state['score']
    
    while (time.time() - start_time) < time_limit:
        iteration += 1
        path = best_final_state['path']
        path_len = len(path)
        if path_len < 10: break
            
        # [V7 Strategy] 80% chance to rollback only last 15-30 steps
        if random.random() < 0.8:
            rollback_steps = random.randint(10, 30)
        else:
            rollback_steps = random.randint(30, max(31, int(path_len * 0.6)))
            
        if rollback_steps >= path_len: rollback_steps = path_len - 2
        cut_start = path_len - rollback_steps
        prefix_path = path[:cut_start]
        
        # Reconstruct state
        temp_map = initial_map_arr.copy()
        prefix_score = 0
        for rect in prefix_path:
            r1, c1, r2, c2 = rect
            s = 0
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if temp_map[r*cols+c] == 1:
                        s += 1
                        temp_map[r*cols+c] = 0
            prefix_score += s
            
        # [V7] Extreme Personality Repair
        repair_weights = weights.copy()
        dice = random.random()
        if dice < 0.4:
            repair_weights['w_island'] = random.randint(150, 300) # Panic
        elif dice < 0.7:
            repair_weights['w_island'] = random.randint(-50, -10) # Chaos
        else:
            repair_weights['w_island'] += random.randint(-20, 20)
            
        huge_beam = int(beam_width * 5.0) 
        
        repaired_state = _run_core_search_logic(
            temp_map, vals_arr, rows, cols, huge_beam, 'omni', 
            prefix_score, prefix_path, repair_weights
        )
        
        if repaired_state['score'] > best_final_state['score']:
            best_final_state = repaired_state
            if best_final_state['score'] > max_score_history:
                max_score_history = best_final_state['score']
                time_limit += 3.0 # Bonus time
        elif repaired_state['score'] == best_final_state['score']:
            if random.random() < 0.3:
                best_final_state = repaired_state

    return {
        'worker_id': seed,
        'score': best_final_state['score'],
        'path': best_final_state['path'],
        'iterations': iteration,
        'personality': personality
    }

class Solver:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def find_first_move(self, grid):
        """
        Find the first valid move in the grid (Greedy approach).
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

    def calculate_simple_score(self, grid):
        """
        Calculate the score using the simple greedy algorithm.
        Returns the total score.
        Optimized to use Numba for performance.
        """
        map_list = []
        val_list = []
        for r in range(self.rows):
            for c in range(self.cols):
                val = grid[r][c]
                val_list.append(val)
                map_list.append(1 if val > 0 else 0)
        
        map_arr = np.array(map_list, dtype=np.int8)
        val_arr = np.array(val_list, dtype=np.int32) # Ensure int32 for sums
        
        return _solve_greedy_numba(map_arr, val_arr, self.rows, self.cols)


    def solve_hydra(self, grid, time_limit=TIME_LIMIT_SECONDS, threads=8):
  
        print(f">> [Solver] Starting Hydra Engine (Limit: {time_limit}s, Threads: {threads})")
        
        # Flatten grid for processing
        map_list = []
        val_list = []
        for r in range(self.rows):
            for c in range(self.cols):
                val = grid[r][c]
                val_list.append(val)
                map_list.append(1 if val > 0 else 0)
        
        beam_width = 30 # Default beam width
        mode = 'god'   # God mode (Classic -> Omni)
        
        tasks = []
        futures = []
        
        base_seed = random.randint(0, 10000)
        
        with ProcessPoolExecutor(max_workers=threads) as executor:
            for i in range(threads):
                personality = {'name': f"Core-{i}"}
                # Distribution of personalities
                if i < 2:
                    personality['w_island'] = 50; personality['w_fragment'] = 2; personality['role'] = 'Balancer'
                elif i < 6:
                    personality['w_island'] = 24; personality['w_fragment'] = 0.5; personality['role'] = 'Striker'
                else:
                    personality['w_island'] = 80; personality['w_fragment'] = 1.0; personality['role'] = 'Heavy'
                
                args = (map_list, val_list, self.rows, self.cols, beam_width, mode, base_seed + i, time_limit, personality)
                futures.append(executor.submit(_solve_process_hydra, args))
            
            best_record = None
            best_score = -1
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    print(f"   [Worker-{res['worker_id'] % 100}] Score: {res['score']} (Iter: {res['iterations']})")
                    if res['score'] > best_score:
                        best_score = res['score']
                        best_record = res
                except Exception as e:
                    print(f"   [Worker Error] {e}")

        if best_record:
            print(f">> [Solver] Best Solution Found: Score {best_score} by {best_record['personality']['role']}")
            return best_record['path'] # Returns list of [r1, c1, r2, c2]
        
        return None
