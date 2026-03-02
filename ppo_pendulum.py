# 导入必要的库
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym

# 仅训练结束后画奖励曲线时使用
import matplotlib.pyplot as plt


# ----- 连续动作策略网络与价值网络 -----
class PolicyModel(nn.Module):
    """连续动作策略模型：输出高斯分布的 mean 与 log_std，用于 Pendulum 等连续动作环境。"""

    def __init__(self, input_dim, action_dim, log_std_init=-0.5):
        super().__init__()
        self.action_dim = action_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        # 可学习的 log_std，与 action_dim 一致
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, x):
        mean = self.fc(x)
        log_std = self.log_std.clamp(-20, 2).expand_as(mean)
        return mean, log_std


class ValueModel(nn.Module):
    """价值模型：给定状态估计价值。"""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.fc(x)


# ----- PPO 算法（连续动作版）-----
class PPO:
    def __init__(
        self,
        env,
        learning_rate=0.001,
        gamma=0.99,
        lamda=0.95,
        clip_eps=0.2,
        epochs=10,
        action_low=None,
        action_high=None,
    ):
        self.env = env
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.action_low = (
            np.array(action_low, dtype=np.float32)
            if action_low is not None
            else np.array(env.action_space.low, dtype=np.float32)
        )
        self.action_high = (
            np.array(action_high, dtype=np.float32)
            if action_high is not None
            else np.array(env.action_space.high, dtype=np.float32)
        )

        self.policy_model = PolicyModel(obs_dim, act_dim).to(self.device)
        self.value_model = ValueModel(obs_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def _get_action_dist(self, state):
        """给定状态，得到高斯分布的 mean、log_std 与 std。"""
        mean, log_std = self.policy_model(state)
        std = torch.exp(log_std)
        return mean, std, log_std

    def choose_action(self, state, deterministic=False):
        """选动作。deterministic=True 时取 mean（用于评估）。返回 (action, log_prob)。"""
        state_tensor = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            mean, std, log_std = self._get_action_dist(state_tensor)

        if deterministic:
            action = mean.cpu().numpy().flatten()
        else:
            dist = torch.distributions.Normal(mean, std)
            x = dist.sample()
            action = x.cpu().numpy().flatten()

        action = np.clip(action, self.action_low, self.action_high)

        # 用于 PPO 的 log_prob：在训练时需存到 buffer，这里用当前 policy 算一次
        action_batch = torch.FloatTensor(np.array([action])).to(self.device)
        mean, std, log_std = self._get_action_dist(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_batch).sum(dim=1).item()

        return action, log_prob

    def save(self, path="ppo_pendulum.pt"):
        """保存策略与价值网络。"""
        torch.save(
            {
                "policy": self.policy_model.state_dict(),
                "value": self.value_model.state_dict(),
            },
            path,
        )

    def load(self, path="ppo_pendulum.pt"):
        """加载策略与价值网络。"""
        data = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(data["policy"])
        self.value_model.load_state_dict(data["value"])

    def calc_advantage(self, td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []
        for r in td_delta[::-1]:
            advantage = r + self.gamma * self.lamda * advantage
            advantage_list.insert(0, advantage)
        return torch.FloatTensor(np.array(advantage_list)).to(self.device)

    def update(self, buffer):
        # buffer: (state, action, reward, next_state, done, log_prob)
        states, actions, rewards, next_states, dones, old_log_probs = zip(*buffer)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).view(-1, 1).to(self.device)

        with torch.no_grad():
            td_target = rewards + (1 - dones) * self.gamma * self.value_model(next_states)
            td_delta = td_target - self.value_model(states)
        old_log_prob = old_log_probs

        advantage = self.calc_advantage(td_delta)
        # 标准化优势，稳定训练、避免 NaN
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.epochs):
            mean, std, _ = self._get_action_dist(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_prob = dist.log_prob(actions).sum(dim=1, keepdim=True)
            ratio = torch.exp(new_log_prob - old_log_prob)
            part1 = ratio * advantage
            part2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -torch.min(part1, part2).mean()
            value_loss = F.mse_loss(self.value_model(states), td_target).mean()

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()


# ----- 训练与可视化 -----
def main(max_episodes=300):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    # Pendulum 每局最多 200 步
    max_steps = 200
    agent = PPO(env, learning_rate=3e-4)
    episode_rewards = []

    for episode in tqdm(range(max_episodes), file=sys.stdout):
        state, _ = env.reset()
        episode_reward = 0
        buffer = []

        for step in range(max_steps):
            action, log_prob = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward, next_state, float(done), log_prob))
            episode_reward += reward
            state = next_state
            if done:
                break

        agent.update(buffer)
        episode_rewards.append(episode_reward)
        if episode % (max_episodes // 10) == 0:
            tqdm.write(f"Episode {episode}: {episode_reward}")

    plt.plot(episode_rewards)
    plt.title("Pendulum-v1 reward")
    plt.show()

    checkpoint_path = "ppo_pendulum.pt"
    agent.save(checkpoint_path)
    print(f"模型已保存到 {checkpoint_path}")

    env.close()
    env_demo = gym.make("Pendulum-v1", render_mode="human")
    _run_demo_episode(env_demo, agent, max_steps=max_steps)
    env_demo.close()


def _run_demo_episode(env, agent, max_steps=200):
    """跑一轮演示。"""
    observation, _ = env.reset()
    for _ in range(max_steps):
        if env.render_mode:
            env.render()
        action, _ = agent.choose_action(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return observation


def run_eval(checkpoint_path="ppo_pendulum.pt", n_episodes=10, render_one=False, max_steps=200):
    """加载已训练模型并评估。"""
    render_mode = "rgb_array" if render_one else None
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    agent = PPO(env)
    if not os.path.isfile(checkpoint_path):
        print(f"未找到模型文件: {checkpoint_path}，请先运行训练: python ppo_pendulum.py")
        env.close()
        return
    agent.load(checkpoint_path)

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action, _ = agent.choose_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)

    env.close()
    mean_reward = sum(rewards) / len(rewards)
    print(f"测试 {n_episodes} 局, 平均回报: {mean_reward:.2f}, 最小: {min(rewards):.2f}, 最大: {max(rewards):.2f}")

    if render_one:
        env = gym.make("Pendulum-v1", render_mode="human")
        agent = PPO(env)
        agent.load(checkpoint_path)
        _run_demo_episode(env, agent, max_steps=max_steps)
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO 训练或评估 (Pendulum-v1)")
    parser.add_argument("--eval", action="store_true", help="仅评估：加载模型并测试，不训练")
    parser.add_argument("--checkpoint", type=str, default="ppo_pendulum.pt", help="模型文件路径")
    parser.add_argument("--episodes", type=int, default=None, help="训练回合数（默认 300）；评估局数（--eval 时默认 10）")
    parser.add_argument("--render", action="store_true", help="评估时额外渲染一局")
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 10 if args.eval else 300

    if args.eval:
        run_eval(checkpoint_path=args.checkpoint, n_episodes=args.episodes, render_one=args.render)
    else:
        main(max_episodes=args.episodes)
