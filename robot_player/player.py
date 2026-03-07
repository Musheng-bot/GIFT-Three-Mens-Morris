import torch

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        
    def get_action(self, state: torch.Tensor) -> tuple[int, int, int]:
        message: str = input(f"Player {self.player_id}, please input your action: ")
        if ":" not in message:
            return (self.player_id, int(message), int(message))
        else:
            original_index, new_index = map(int, message.split(":"))
            return (self.player_id, original_index, new_index)
