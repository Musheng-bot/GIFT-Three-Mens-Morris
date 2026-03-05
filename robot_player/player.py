from board import Board

class Player:
    def __init__(self, player_id: int, board: Board):
        self.player_id = player_id
        self.board = board
        
    def get_action(self) -> str:
        message: str = input(f"Player {self.player_id}, please input your action: ")
        return message