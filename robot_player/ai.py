import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from player import Player

class QNet(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 256, output_size: int = 81):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # zip(*batch) 解压为 (states, actions, rewards, next_states, dones)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class AIPlayer(Player):
    def __init__(
        self,
        player_id,
        batch_size: int = 64,
        lr=1e-3,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update=100,
    ):
        super().__init__(player_id)

        self.policy_net = QNet()
        self.target_net = QNet()
        # 初始化目标网络权重
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不参与梯度更新

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = batch_size
        self.gamma = gamma

        # 探索率参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update

        self.steps_done = 0
        self.player_id = player_id  # 确保保存 player_id (0 or 1)

    def get_legal_action_mask(self, state: torch.Tensor) -> torch.Tensor:
        """
        生成合法动作掩码 (0-80)。
        规则：
        1. 放置阶段 (棋子<3): 必须 src == dst, 且目标位置为空。
        2. 移动阶段 (棋子>=3): 必须 src != dst, src 是己方棋子, dst 为空。任意移动。
        """
        mask = torch.zeros(size=(81,), dtype=bool)

        my_pieces = torch.sum(state == 1)
        is_placing_phase = (my_pieces < 3)

        for i in range(9):  # src
            for j in range(9):  # dst
                idx = i * 9 + j
                
                if is_placing_phase:
                    if i == j and state[j] == 0:
                        mask[idx] = True
                    else:
                        mask[idx] = False
                else:
                    if i != j and state[i] == 1 and state[j] == 0:
                        mask[idx] = True
                    else:
                        mask[idx] = False
        return mask

    def get_action(self, state: torch.Tensor) -> tuple[int, int, int]:
        legal_mask = self.get_legal_action_mask(state)

        self.steps_done += 1
        if random.random() < self.epsilon:
            # 探索：从合法动作中随机选一个
            legal_indices = np.where(legal_mask)[0]
            action_idx = random.choice(legal_indices)
        else:
            # 利用：神经网络预测
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0)).squeeze(0).cpu().numpy()

                # 关键：将非法动作的 Q 值设为负无穷，防止被选中
                q_values[~legal_mask] = -float("inf")
                action_idx = np.argmax(q_values)

        # 衰减 epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 解码动作: idx -> src, dst
        src = action_idx // 9
        dst = action_idx % 9

        return (self.player_id, src, dst)

    def update(self, state: torch.Tensor, action: tuple[int, int, int], reward: float, next_state: torch.Tensor, done: bool):
        """
        state, next_state: 建议传入 numpy 数组或 list，内部转 tensor
        action: tuple (pid, src, dst)
        """

        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为 Tensor
        states = torch.FloatTensor(np.stack(states))
        actions = torch.LongTensor(actions)  # 动作索引
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(torch.stack(next_states))
        dones = torch.FloatTensor(dones)

        policy: torch.Tensor = self.policy_net(states)
        current_q_values = (
            policy.gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            next_q_values = self.target_net(next_states)

            # --- 关键：对下一状态的非法动作进行 Mask ---
            # 我们需要对 batch 中的每一个样本分别构建 mask
            max_next_q = torch.zeros(self.batch_size)

            for i in range(self.batch_size):
                ns_np = next_states[i]
                # 如果游戏已结束，target Q 就是 0 (因为 dones=1 时后面会乘 0)
                if dones[i]:
                    max_next_q[i] = 0.0
                    continue

                # 构建该样本的合法掩码
                legal_mask = self.get_legal_action_mask(ns_np)
                q_vals = next_q_values[i].cpu().numpy()

                if not np.any(legal_mask):
                    max_next_q[i] = 0.0
                else:
                    q_vals[~legal_mask] = -float("inf")
                    max_next_q[i] = np.max(q_vals)

            max_next_q = torch.FloatTensor(max_next_q)

            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

