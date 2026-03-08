import numpy as np
import torch

class Board:
    def __init__(self, board: np.ndarray, player_id1: int, player_id2: int):
        self.value = board # 1D array
        self.player_id1 = player_id1
        self.player_id2 = player_id2

    def get_state(self, player_id: int) -> torch.Tensor:
        # 使用 copy() 确保不影响原数组
        b_copy = self.value.copy()
        state = np.zeros(9, dtype=np.float32)

        # 逻辑归一化：1 是自己，-1 是对手
        state[b_copy == player_id] = 1.0
        # 找到对手的 ID (假设 player_id 为 1, 则对手为 2; 反之亦然)
        # 或者更通用的做法：
        other_mask = (b_copy != player_id) & (b_copy != 0)
        state[other_mask] = -1.0

        return torch.from_numpy(state).clone()  # 确保内存独立

    def apply_action(self, action: tuple[int, int, int]):
        player_id, original_index, new_index = action
        if original_index == new_index:
            if self.value[original_index] != 0:
                raise ValueError(f"Invalid action: The position {original_index} should be empty")
            else:
                self.value[new_index] = player_id
        else:
            if self.value[original_index] != player_id or self.value[new_index] != 0:
                raise ValueError(f"Invalid action: The position {original_index} is not your piece or the position {new_index} is already occupied")
            self.value[original_index] = 0
            self.value[new_index] = player_id
        
    def is_nearby(self, index1: int, index2: int) -> bool:
        return abs(index1 - index2) == 1 or abs(index1 - index2) == 3
    
    def get_winner(self) -> int:
        b = self.value
        # 8种获胜组合：3行 + 3列 + 2对角线
        wins = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # 行
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # 列
            (0, 4, 8), (2, 4, 6)             # 对角线
        ]
        
        for x, y, z in wins:
            if b[x] != 0 and b[x] == b[y] == b[z]:
                return b[x]
        return 0

    
    def __repr__(self):
        new_array = self.value.reshape(3, -1)
        result = ""
        result += "---------\n"
        for row in new_array:
            row_str = " | ".join(map(str, row))
            result += row_str + "\n"
            result += "---------\n"
        return result
    
    def get_reward(self, player_id: int) -> float:
        # 1. 获取归一化状态：1 是自己，-1 是对手
        state = self.get_state(player_id) 
        
        winner = self.get_winner()
        if winner == player_id:
            return 100.0
        if winner != 0: # 说明对手赢了
            return -100.0

        # 2. 基于归一化状态的中间奖励
        mid_reward = 0.0
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        
        for x, y, z in wins:
            line = [state[x].item(), state[y].item(), state[z].item()]
            my_count = line.count(1.0)
            opp_count = line.count(-1.0)
            empty_count = line.count(0.0)
            
            if my_count == 2 and empty_count == 1:
                mid_reward += 10.0  # 离胜利只差一步
            if opp_count == 2 and empty_count == 1:
                mid_reward += 15.0  # 成功堵截对手的关键位（防守奖励通常设高一点）

        return mid_reward - 1.0 # 步数惩罚，逼迫 AI 尽快获胜
        
    def clear(self):
        self.value = np.zeros(shape=(9,), dtype=int)
        