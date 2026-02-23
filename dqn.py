import argparse
import random
from collections import deque
from typing import Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    """Q网络，用于近似Q值函数"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """存储一个经验元组"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """随机采样一个批次的经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        target_update_freq: int = 1000,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device

        # 创建Q网络和目标网络
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_capacity)

        # 训练步数计数器
        self.steps = 0

    def select_action(self, state: np.ndarray) -> int:
        """使用epsilon-greedy策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机探索
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()  # 选择Q值最大的动作

    def update(self) -> float:
        """执行一次网络更新，返回损失值"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor).squeeze()

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """保存模型权重"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)

    def load(self, path: str):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


def train(
    env_name: str = "CartPole-v1",
    num_episodes: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    target_update_freq: int = 1000,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    eval_interval: int = 10,
    save_path: str = "dqn_model.pth",
    device: str = "cpu",
):
    """训练DQN智能体"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        target_update_freq=target_update_freq,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        device=device,
    )

    episode_rewards = []
    episode_losses = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        step_count = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.buffer.push(state, action, reward, next_state, done)

            # 更新网络
            loss = agent.update()
            total_loss += loss if loss else 0

            state = next_state
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / step_count if step_count > 0 else 0)

        # 输出训练信息
        if episode % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_loss = np.mean(episode_losses[-eval_interval:])
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Loss: {avg_loss:7.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.buffer):5d}")

    # 训练完成，保存模型
    agent.save(save_path)
    print(f"Training completed. Model saved to {save_path}")
    env.close()

    return agent, episode_rewards, episode_losses


def evaluate(
    env_name: str = "CartPole-v1",
    model_path: str = "dqn_model.pth",
    num_episodes: int = 10,
    render: bool = True,
    device: str = "cpu",
):
    """评估训练好的DQN智能体"""
    # Gymnasium 需在创建环境时指定 render_mode 才能显示图形界面
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建智能体并加载模型
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon=0.0,  # 评估时不探索
        device=device,
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # 确保评估时不探索

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.q_network(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Evaluation Episode {episode:3d} | Reward: {total_reward:7.2f}")

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nEvaluation completed over {num_episodes} episodes.")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="DQN 强化学习训练与评估")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="模式：train (训练) 或 eval (评估)")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gym 环境名称 (默认: CartPole-v1)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="训练回合数 (默认: 1000)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率 (默认: 0.001)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子 (默认: 0.99)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="初始探索率 (默认: 1.0)")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="探索率衰减率 (默认: 0.995)")
    parser.add_argument("--epsilon_min", type=float, default=0.01,
                        help="最小探索率 (默认: 0.01)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="目标网络更新频率 (默认: 1000步)")
    parser.add_argument("--buffer_capacity", type=int, default=10000,
                        help="经验回放缓冲区容量 (默认: 10000)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="训练批次大小 (默认: 64)")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="训练中评估间隔 (默认: 10回合)")
    parser.add_argument("--save_path", type=str, default="dqn_model.pth",
                        help="模型保存路径 (默认: dqn_model.pth)")
    parser.add_argument("--load_path", type=str, default="dqn_model.pth",
                        help="模型加载路径 (默认: dqn_model.pth)")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                        help="评估回合数 (默认: 10)")
    parser.add_argument("--no_render", action="store_true",
                        help="评估时不渲染环境")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备 (默认: cuda 如果可用，否则 cpu)")

    args = parser.parse_args()

    if args.mode == "train":
        print(f"开始训练 DQN 于环境 {args.env}")
        print(f"设备: {args.device}")
        print(f"超参数: lr={args.lr}, gamma={args.gamma}, epsilon={args.epsilon}, "
              f"epsilon_decay={args.epsilon_decay}, epsilon_min={args.epsilon_min}")
        print(f"         target_update_freq={args.target_update_freq}, "
              f"buffer_capacity={args.buffer_capacity}, batch_size={args.batch_size}")
        print("-" * 60)

        train(
            env_name=args.env,
            num_episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            target_update_freq=args.target_update_freq,
            buffer_capacity=args.buffer_capacity,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            save_path=args.save_path,
            device=args.device,
        )

    elif args.mode == "eval":
        print(f"开始评估 DQN 于环境 {args.env}")
        print(f"设备: {args.device}")
        print(f"加载模型: {args.load_path}")
        print("-" * 60)

        evaluate(
            env_name=args.env,
            model_path=args.load_path,
            num_episodes=args.num_eval_episodes,
            render=not args.no_render,
            device=args.device,
        )


if __name__ == "__main__":
    main()