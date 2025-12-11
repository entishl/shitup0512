import sys
import os
import numpy as np

# Apply path patch to ensure we can import from src correctly when running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NOTE: Do not import GameAutomator at module level to allow lightweight testing
# from src.automator import GameAutomator 

def get_partitions_of_k_target_n(k, n, min_val=1):
    """
    Find all combinations of k numbers that sum to n.
    Used for precomputing or on-the-fly generation if needed.
    However, for our case, we just need "any combination summing to target".
    """
    pass

class ViabilitySolver:
    def __init__(self):
        # Precompute simple partitions for numbers 1-9 to make 10
        # actually, we just need to know: for a number N, what sums to 10-N?
        # Target = 10-N.
        pass

    @staticmethod
    def _find_sum_combinations(target, current_combo, start_val, results):
        if target == 0:
            results.append(list(current_combo))
            return
        if target < 0:
            return
        
        for val in range(start_val, target + 1):
            # Optimization: We only have digits 1-9 in the grid
            if val > 9: break 
            current_combo.append(val)
            ViabilitySolver._find_sum_combinations(target - val, current_combo, val, results)
            current_combo.pop()

    @staticmethod
    def get_combinations_for_target(target):
        """
        Returns a list of combinations (lists of ints) that sum to target.
        e.g. target=3 -> [[1, 1, 1], [1, 2], [3]]
        Sorted by length (greedy preference: try to use fewer items? or more items?)
        Actually, for 10-sum game, usually using larger items first is better specifically for clearing?
        But verifying solvability: getting rid of "hard to match" numbers is priority.
        Hard numbers are usually the ones that require specific complements.
        """
        results = []
        ViabilitySolver._find_sum_combinations(target, [], 1, results)
        # Sort results? 
        # Strategy: Try to match with "hardest to satisfy" numbers first? 
        # Or just try all. Backtracking will cover it.
        # Sorting by reverse length (largest numbers first) might be faster for finding A solution.
        results.sort(key=lambda x: len(x)) 
        return results

def solve_backtrack(counts):
    # Base case: if all counts are 0, success
    if all(c == 0 for c in counts.values()):
        return True

    # Pick the largest available number to try to clear
    # Why largest? Because they have fewer combinations to sum to 10.
    # e.g. 9 needs 1. 8 needs 2 (or 1+1). 1 needs 9 (or 8+1, 7+2...)
    start_num = -1
    for i in range(9, 0, -1):
        if counts[i] > 0:
            start_num = i
            break
            
    if start_num == -1: 
        # Should be caught by base case, but if we have 0s? 
        # The grid might have 0s (empty/unrecognized). We ignore them.
        return True

    # We have a number 'start_num'. We need 'needed = 10 - start_num'.
    needed = 10 - start_num
    
    # Get all combinations of available numbers that sum to 'needed'
    # We generate these on the fly or use a helper. 
    # Since 'needed' is small (1..9), this is fast.
    combinations = ViabilitySolver.get_combinations_for_target(needed)
    
    # Try each combination
    for combo in combinations:
        # Check if we have enough items for this combo
        # combo is a list, e.g. [1, 2]
        
        # Check availability
        possible = True
        temp_counts = counts.copy() # Shallow copy is fine for dict of ints
        
        # Decrement the primary number
        temp_counts[start_num] -= 1
        
        for req in combo:
            if temp_counts.get(req, 0) > 0:
                temp_counts[req] -= 1
            else:
                possible = False
                break
        
        if possible:
            # Recurse
            if solve_backtrack(temp_counts):
                return True
                
    return False

def is_numerically_possible(grid):
    """
    Check if the grid numbers can be theoretically partitioned into groups summing to 10.
    """
    # 1. Count numbers
    counts = {i: 0 for i in range(1, 10)}
    unique, quantities = np.unique(grid, return_counts=True)
    for u, q in zip(unique, quantities):
        if u > 0 and u <= 9: # Ignore 0 and potential garbage
            counts[u] = q
            
    total_items = sum(counts.values())
    print(f"当前数字统计: {counts}")
    print(f"有效数字总数: {total_items}")
    
    # Basic logic check: Sum must be divisible by 10
    total_sum = sum(k * v for k, v in counts.items())
    if total_sum % 10 != 0:
        print(f"[FAIL] 总和 {total_sum} 不能被 10 整除")
        return False

    # 2. Backtracking Validation
    return solve_backtrack(counts)

def main():
    print("=== Solver Viability Debug Tool ===")
    
    try:
        from src.automator import GameAutomator
        
        # Initialize Automator (connects to window)
        automator = GameAutomator()
        
        # Capture screen and recognize
        automator.capture_and_recognize()
        grid = automator.grid
        
        print("\n=== Grid State ===")
        print(grid)
        
        # Run Check
        if is_numerically_possible(grid):
            print(">>> [PASS] 理论存在解 (Backtracking Check Passed).")
        else:
            print(">>> [FAIL] 理论无解 (Backtracking Check Failed).")
            
    except ImportError:
        print("Error: Could not import GameAutomator. Make sure dependencies (cv2) are installed.")
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nDone.")

if __name__ == "__main__":
    main()
