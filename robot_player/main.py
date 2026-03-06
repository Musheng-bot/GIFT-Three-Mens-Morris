import numpy as np
from ai import AIPlayer
from board import Board
from player import Player

def init(chess_board: np.ndarray, player_id1: int, player_id2: int, use_ai: bool = False, is_train: bool = False, ai_player: AIPlayer | None = None) -> tuple[Board, Player, Player]:
    board = Board(chess_board, player_id1, player_id2)
    player1, player2 = Player(player_id1, board), Player(player_id2, board)
    if use_ai or is_train:
        player2 = AIPlayer(player_id2, board) if ai_player is None else ai_player
        player2.board = board
    if is_train:
        player1 = AIPlayer(player_id1, board)
    return board, player1, player2


def play(use_ai: bool = True, ai_player: AIPlayer | None = None):
    chess_board = np.zeros(shape=(9,), dtype=int)
    board, player1, player2 = init(chess_board, 1, 2, use_ai=use_ai, is_train=False, ai_player=ai_player)
    current_player = player1
    while board.get_winner() == 0:
        try:
            print(f"Current board state:\n{board}")
            action = f"{current_player.player_id}_{current_player.get_action()}"
            print("Current action:", action)
            try:
                board.apply_action(action)
            except ValueError as e:
                print(e)
                continue
            current_player = player2 if current_player == player1 else player1
            print("-----------------\n")
        except IndexError as e:
            print(e)
            continue
    winner = board.get_winner()
    print(board)
    print(f"Winner: Player {winner}")
    
def train_ai(episodes: int = 10000) -> tuple[AIPlayer, AIPlayer]:
    chess_board = np.zeros(shape=(9,), dtype=int)
    board, player1, player2 = init(chess_board, 1, 2, is_train=True)
    
    player1_reward = 0
    player2_reward = 0
    for ep in range(episodes):
        board.clear()
        current_player = player1
        player1_reward = 0
        player2_reward = 0

        old_state = board.get_state()
        action = f"{current_player.player_id}_{current_player.get_action()}"
        reward = board.get_reward(current_player.player_id)
        
        while board.get_winner() == 0:
            try:
                old_state = board.get_state()
                action = f"{current_player.player_id}_{current_player.get_action()}"
                try:
                    board.apply_action(action)
                except ValueError as e:
                    print(e)
                    continue
                reward = board.get_reward(current_player.player_id)
                if isinstance(current_player, AIPlayer):
                    new_state = board.get_state()
                    current_player.update_q_table(
                        old_state,
                        action,
                        reward,
                        new_state,
                    )
                if current_player == player1:
                    player1_reward += reward
                else:
                    player2_reward += reward
                current_player = player2 if current_player == player1 else player1
                if isinstance(current_player, AIPlayer):
                    current_player.reduce_epsilon()
            except IndexError as e:
                print(e)
                continue
        reward = board.get_reward(current_player.player_id)
        current_player.update_q_table(old_state, action, reward, new_state)
        if current_player == player1:
            player1_reward += reward
        else:
            player2_reward += reward
        if ep % 1000 == 0:
            print(f"Episode {ep}: Player 1 reward = {player1_reward}, Player 2 reward = {player2_reward}")
            if isinstance(player1, AIPlayer):
                player1.reset_epsilon()
            if isinstance(player2, AIPlayer):
                player2.reset_epsilon()



    return player1, player2
    
def get_board_state(board: Board) -> str:
    return board.get_state()

def save_ai_player(ai_player: AIPlayer, filename: str):
    np.save(filename, ai_player.q_table)
    
def load_ai_player(filename: str) -> AIPlayer:
    q_table = np.load(filename, allow_pickle=True)
    ai_player = AIPlayer(player_id=2, board=None)
    ai_player.q_table = q_table
    return ai_player
        
def train_and_play(episodes: int = 10000):
    _, ai_player2 = train_ai(episodes)
    save_ai_player(ai_player2, "ai_player2.npy")
    play(use_ai=True, ai_player=ai_player2)
    
def play_with_loaded_ai(filename: str):
    ai_player2 = load_ai_player(filename)
    play(use_ai=True, ai_player=ai_player2)
        
if __name__ == "__main__":
    play_with_loaded_ai("ai_player2.npy")
