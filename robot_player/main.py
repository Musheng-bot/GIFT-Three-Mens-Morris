import numpy as np
from ai import AIPlayer
from board import Board
from player import Player

def init(chess_board: np.ndarray, player_id1: int, player_id2: int, use_ai: bool = False) -> tuple[Board, Player, Player]:
    board = Board(chess_board, player_id1, player_id2)
    player1, player2 = Player(player_id1, board), Player(player_id2, board)
    if use_ai:
        player2 = AIPlayer(player_id2, board)
    return board, player1, player2


def play(use_ai: bool = True):
    chess_board = np.zeros(shape=(9,), dtype=int)
    board, player1, player2 = init(chess_board, 1, 2, use_ai)
    current_player = player1
    while board.get_winner() == 0:
        try:
            print(f"Current board state:\n{board}")
            old_state = board.get_state()
            action = f"{current_player.player_id}_{current_player.get_action()}"
            print("Current action:", action)
            try:
                board.apply_action(action)
            except ValueError as e:
                print(e)
                continue
            if use_ai and current_player == player2:
                new_state = board.get_state()
                player2.update_q_table(old_state, action, board.get_reward(player2.player_id), new_state)
            current_player = player2 if current_player == player1 else player1
            print("-----------------\n")
        except IndexError as e:
            print(e)
            continue
    winner = board.get_winner()
    print(board)
    print(f"Winner: Player {winner}")
    
def train_ai():
    pass
    
def get_board_state(board: Board) -> str:
    return board.get_state()
        
if __name__ == "__main__":
    play(False)
