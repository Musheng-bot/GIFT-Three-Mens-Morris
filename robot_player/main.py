import numpy as np
from ai import AIPlayer
from board import Board
from player import Player
import torch

# def play(use_ai: bool = True, ai_player: AIPlayer | None = None):
#     chess_board = np.zeros(shape=(9,), dtype=int)
#     board, player1, player2 = init(chess_board, 1, 2, use_ai=use_ai, is_train=False, ai_player=ai_player)
#     current_player = player1
#     while board.get_winner() == 0:
#         try:
#             print(board)
#             state = board.get_state(current_player.player_id)
#             action = current_player.get_action(state)
#             try:
#                 board.apply_action(action)
#             except ValueError as e:
#                 print(e)
#                 continue
#             current_player = player2 if current_player == player1 else player1
#             print("-----------------\n")
#         except IndexError as e:
#             print(e)
#             continue
#     winner = board.get_winner()
#     print(board)
#     print(f"Winner: Player {winner}")
    
def switch_player(current_player: Player, player1: Player, player2: Player) -> Player:
    return player2 if current_player == player1 else player1
    
def train_ai(episodes: int = 10000) -> AIPlayer:
    # 初始化两个相同的 AI
    player1 = AIPlayer(player_id=1)
    player2 = AIPlayer(player_id=2)
    board = Board(
        np.zeros(shape=(9,), dtype=int), 
        player_id1=player1.player_id, 
        player_id2=player2.player_id
    )


    # 让 player2 共享 player1 的网络，或者定期同步
    player2.policy_net = player1.policy_net

    for episode in range(episodes):
        board.clear()
        state = board.get_state(player_id=1)

        while not board.get_winner():
            # --- 玩家 1 回合 ---
            action = player1.get_action(state)
            board.apply_action(action)
            reward = board.get_reward(player_id=1)
            next_state = board.get_state(player_id=1)
            done = board.get_winner() != 0

            player1.update(state, action, reward, next_state, done)

            if done:
                break

            # --- 玩家 2 回合 (同样更新 player1 的大脑) ---
            state2 = board.get_state(player_id=2)
            action2 = player2.get_action(state2)  # 其实是在用同一个网络
            board.apply_action(action2)
            # 这里的 reward 是针对玩家 2 的
            reward2 = board.get_reward(player_id=2)
            next_state2 = board.get_state(player_id=2)

            # 关键：所有人的经验都喂给同一个大脑
            player1.update(state2, action2, reward2, next_state2, board.get_winner() != 0)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Player 1 Win Rate: {player1.win_rate:.2f}")
    return player1

def save_ai_player(ai_player: AIPlayer, filename: str):
    torch.save(ai_player.policy_net.state_dict(), filename)
    torch.save(ai_player.target_net.state_dict(), filename.replace(".pt", "_target.pt"))
    
def load_ai_player(filename: str) -> AIPlayer:
    ai_player = AIPlayer(player_id=2)
    ai_player.policy_net.load_state_dict(torch.load(filename))
    ai_player.target_net.load_state_dict(torch.load(filename.replace(".pt", "_target.pt")))
    return ai_player
    
if __name__ == "__main__":
    player = train_ai(episodes=20000)
    save_ai_player(player, "ai_player1.pt")
