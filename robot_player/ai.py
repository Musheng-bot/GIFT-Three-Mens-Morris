import numpy as np
import random
from board import Board
from player import Player

class AIPlayer(Player):
    def __init__(self, player_id, board: Board, epsilon=0.1, alpha=0.1, gamma=0.9):
        super().__init__(player_id, board)
        
        self.q_table: dict[str, dict[str, float]] = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
                
    def get_valid_actions(self, state):
        zero_count = np.sum(self.board.value.flatten() == 0)
        if zero_count > 3:
            return list(map(lambda x: f"{str(x)}", np.argwhere(self.board.value.flatten() == 0).flatten()))
        elif zero_count < 3:
            raise ValueError("Invalid state")
        else:
            zero_indecies = np.argwhere(self.board.value.flatten() == 0).flatten()
            player_indecies = np.argwhere(self.board.value.flatten() == self.player_id).flatten()
            return [f"{str(b)}:{str(a)}" for a in zero_indecies for b in player_indecies]
        
    def choose_action(self, state):
        valid_actions = self.get_valid_actions(state)
        print(f"Player {self.player_id} valid actions:", valid_actions)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            if state not in self.q_table:
                self.q_table[state] = {}
            for action in valid_actions:
                self.q_table[state][action] = 0
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [valid_actions[i] for i in range(len(valid_actions)) if q_values[i] == max_q]
            return random.choice(best_actions)
        
    def update_q_table(self, state: str, action: str, reward: float, next_state: str):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
            
        error = max(self.q_table.get(next_state, {}).values(), default=0) - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * (reward + self.gamma * error)

    def get_action(self):
        return self.choose_action(self.board.get_state())
