import numpy as np
from ai import AIPlayer
from board import Board
from player import Player
import torch

def init(chess_board: np.ndarray, player_id1: int, player_id2: int, use_ai: bool = False, is_train: bool = False, ai_player: AIPlayer | None = None) -> tuple[Board, Player, Player]:
    board = Board(chess_board, player_id1, player_id2)
    player1, player2 = Player(player_id1), Player(player_id2)
    if use_ai or is_train:
        player2 = AIPlayer(player_id2) if ai_player is None else ai_player
    if is_train:
        player1 = AIPlayer(player_id1)
    return board, player1, player2


def play(use_ai: bool = True, ai_player: AIPlayer | None = None):
    chess_board = np.zeros(shape=(9,), dtype=int)
    board, player1, player2 = init(chess_board, 1, 2, use_ai=use_ai, is_train=False, ai_player=ai_player)
    current_player = player1
    while board.get_winner() == 0:
        try:
            state = board.get_state(current_player.player_id)
            action = current_player.get_action(state)
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
    
def switch_player(current_player: Player, player1: Player, player2: Player) -> Player:
    return player2 if current_player == player1 else player1
    
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
        
        done = False
        while True:
            state = board.get_state(current_player.player_id)
            action = current_player.get_action(state)
            done = board.get_winner() != 0
            reward = board.get_reward(current_player.player_id)
            board.apply_action(action)
            new_state = board.get_state(current_player.player_id)
            if isinstance(current_player, AIPlayer):
                current_player.update(state, action, reward, new_state, done)
            state = new_state
            
            current_player = switch_player(current_player, player1, player2)
            if current_player == player1:
                player1_reward += reward
            else:
                player2_reward += reward
            
            if done:
                current_player = switch_player(current_player, player1, player2)
                reward = board.get_reward(current_player.player_id)
                current_player.update(state, action, reward, new_state, done)
                break

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}: Player 1 reward = {player1_reward}, Player 2 reward = {player2_reward}")
    return player1, player2
    
def get_board_state(board: Board) -> str:
    return board.get_state()

def save_ai_player(ai_player: AIPlayer, filename: str):
    torch.save(ai_player.policy_net.state_dict(), filename)
    torch.save(ai_player.target_net.state_dict(), filename.replace(".pt", "_target.pt"))
    
def load_ai_player(filename: str) -> AIPlayer:
    ai_player = AIPlayer(player_id=2, board=None)
    ai_player.policy_net.load_state_dict(torch.load(filename))
    ai_player.target_net.load_state_dict(torch.load(filename.replace(".pt", "_target.pt")))
    return ai_player
        
def train_and_play(episodes: int = 10000):
    _, ai_player2 = train_ai(episodes)
    save_ai_player(ai_player2, "ai_player2.pt")
    play(use_ai=True, ai_player=ai_player2)
    
def play_with_loaded_ai(filename: str):
    ai_player2 = load_ai_player(filename)
    play(use_ai=True, ai_player=ai_player2)
        
if __name__ == "__main__":
    train_and_play()
