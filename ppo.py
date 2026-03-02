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


# ----- 策略网络与价值网络 -----
class PolicyModel(nn.Module):
    """策略模型：给定状态输出各动作概率分布。"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.fc(x)


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


# ----- PPO 算法 -----
class PPO:
    def __init__(
        self,
        env,
        learning_rate=0.001,
        gamma=0.99,
        lamda=0.95,
        clip_eps=0.2,
        epochs=10,
    ):
        self.env = env
        # 折扣因子
        self.gamma = gamma
        # 优势函数 在 GAE (Generalized Advantage Estimation) 里控制“用多长的回报估计”来算优势。
        self.lamda = lamda
        # PPO裁剪范围
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.policy_model = PolicyModel(obs_dim, act_dim).to(self.device)
        self.value_model = ValueModel(obs_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def choose_action(self, state, deterministic=False):
        """选动作。deterministic=True 时取概率最大的动作（用于评估）。"""
        state = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            action_prob = self.policy_model(state)
        
        # 评价或者演示的时候，取概率最大的动作
        if deterministic:
            action = action_prob.argmax(dim=1).item()
        # 训练的时候，取概率分布中的动作
        else:
            c = torch.distributions.Categorical(action_prob)
            action = c.sample().item()
        return action

    def save(self, path="ppo_acrobot.pt"):
        """保存策略与价值网络。"""
        torch.save(
            {
                "policy": self.policy_model.state_dict(),
                "value": self.value_model.state_dict(),
            },
            path,
        )

    def load(self, path="ppo_acrobot.pt"):
        """加载策略与价值网络（需在创建 PPO 后、与当前 env 一致时调用）。"""
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
        states, actions, rewards, next_states, dones = zip(*buffer)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)
        
        # 用当前策略算一次 log π_old(a|s) 并固定下来，后面多轮更新时都拿它当“旧策略”
        with torch.no_grad():
            old_action_log_prob = torch.log(self.policy_model(states).gather(1, actions))
            td_target = rewards + (1 - dones) * self.gamma * self.value_model(next_states)
            #td 是 Temporal Difference（时序差分）的缩写
            td_delta = td_target - self.value_model(states)

        advantage = self.calc_advantage(td_delta)

        for _ in range(self.epochs):
            # 当前（更新中）策略的 log π(a|s)
            action_log_prob = torch.log(self.policy_model(states).gather(1, actions))
            # 计算重要性比率：新策略与旧策略的比值
            ratio = torch.exp(action_log_prob - old_action_log_prob)
            # ratio > 1：新策略对该动作赋予更高概率；结合 A>0 时是期望方向，A<0 时则过度更新
            # ratio < 1：新策略对该动作赋予更低概率；结合 A<0 时是期望方向，A>0 时则过度更新
            # clip 限制 ratio 在 [1-ε, 1+ε]，避免单次更新步长过大、策略偏离旧数据分布过远

            # 未裁剪的重要性采样策略梯度项
            part1 = ratio * advantage
            
            #对 ratio 做 PPO 的 clip 后再乘 advantage
            part2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            
            #当 ratio 偏离 1 太多时，part1 会变大或变小得很厉害，而 part2 被 clip 住；取 min 会选更保守的那一项，从而限制更新幅度，这就是 PPO-Clip 的“悲观”更新。
            
            #PPO-Clip 目标（取 min 再取负做最小化）
            policy_loss = -torch.min(part1, part2).mean()
            value_loss = F.mse_loss(self.value_model(states), td_target).mean()

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()


# ----- 训练与可视化 -----
def main(max_episodes=300):
    # 创建环境（脚本中可用 'human' 看窗口，或 'rgb_array' 配合 GymHelper）
    env = gym.make("Acrobot-v1", render_mode="rgb_array")

    max_steps = 500
    agent = PPO(env)
    episode_rewards = []

    for episode in tqdm(range(max_episodes), file=sys.stdout):
        state, _ = env.reset()
        episode_reward = 0
        buffer = []

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward, next_state, float(done)))
            episode_reward += reward
            state = next_state
            if done:
                break

        agent.update(buffer)
        episode_rewards.append(episode_reward)
        if episode % (max_episodes // 10) == 0:
            tqdm.write(f"Episode {episode}: {episode_reward}")

    plt.plot(episode_rewards)
    plt.title("reward")
    plt.show()

    # 保存模型，便于之后单独测试
    checkpoint_path = "ppo_acrobot.pt"
    agent.save(checkpoint_path)
    print(f"模型已保存到 {checkpoint_path}")

    # 演示一轮（另开 human 窗口，与 bak/ppo_tutorial.py 一致）
    env.close()
    env_demo = gym.make("Acrobot-v1", render_mode="human")
    _run_demo_episode(env_demo, agent, max_steps=max_steps)
    env_demo.close()


def _run_demo_episode(env, agent, max_steps=500):
    """跑一轮；若 env 为 human 渲染则直接调用 env.render() 弹窗。"""
    observation, _ = env.reset()
    for _ in range(max_steps):
        if env.render_mode:
            env.render()
        action = agent.choose_action(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return observation


def run_eval(checkpoint_path="ppo_acrobot.pt", n_episodes=10, render_one=False, max_steps=500):
    """加载已训练模型并评估：跑 n_episodes 局，打印平均回报；可选渲染一局。"""
    render_mode = "rgb_array" if render_one else None
    env = gym.make("Acrobot-v1", render_mode=render_mode)
    agent = PPO(env)
    if not os.path.isfile(checkpoint_path):
        print(f"未找到模型文件: {checkpoint_path}，请先运行训练: python ppo.py")
        env.close()
        return
    agent.load(checkpoint_path)

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = agent.choose_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)

    env.close()
    mean_reward = sum(rewards) / len(rewards)
    print(f"测试 {n_episodes} 局, 平均回报: {mean_reward:.1f}, 最小: {min(rewards):.1f}, 最大: {max(rewards):.1f}")

    if render_one:
        env = gym.make("Acrobot-v1", render_mode="human")
        agent = PPO(env)
        agent.load(checkpoint_path)
        _run_demo_episode(env, agent, max_steps=max_steps)
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO 训练或评估")
    parser.add_argument("--eval", action="store_true", help="仅评估：加载模型并测试，不训练")
    parser.add_argument("--checkpoint", type=str, default="ppo_acrobot.pt", help="模型文件路径")
    parser.add_argument("--episodes", type=int, default=None, help="训练回合数（训练模式，默认 300）；评估局数（--eval，默认 10）")
    parser.add_argument("--render", action="store_true", help="评估时额外渲染一局")
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 10 if args.eval else 300

    if args.eval:
        run_eval(checkpoint_path=args.checkpoint, n_episodes=args.episodes, render_one=args.render)
    else:
        main(max_episodes=args.episodes)