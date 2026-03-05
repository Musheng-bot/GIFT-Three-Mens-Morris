import numpy as np


class Board:
    def __init__(self, board: np.ndarray, player_id1: int, player_id2: int):
        self.value = board # 1D array
        self.player_id1 = player_id1
        self.player_id2 = player_id2

    def get_state(self):
        return "".join(map(str, self.value))
    
    def apply_action(self, action: str):
        if ":" not in action:
            zero_count = np.sum(self.value == 0).item()
            if zero_count <= 3:
                raise ValueError("Invalid action: You should not place more than 3 pieces")
            player_id, index = action.split("_")
            index = int(index)
            if self.value[index] != 0:
                raise ValueError(f"Invalid action: The position {index} is already occupied")
            self.value[index] = int(player_id)
        else:
            player_id, indecies = action.split("_")
            original_index, new_index = map(int, indecies.split(":"))
            if self.value[original_index] != int(player_id) or self.value[new_index] != 0:
                raise ValueError(f"Invalid action: The position {original_index} is not your piece or the position {new_index} is already occupied")
            self.value[original_index] = 0
            self.value[new_index] = int(player_id)
            
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
        winner = self.get_winner()
        if winner == player_id:
            return 100
        elif winner == 0:
            return -1
        else:
            return -100