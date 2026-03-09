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
from torch.distributions import Categorical


class PolicyModel(nn.Module):
    """
    策略网络：输入状态，输出动作概率分布
    input_dim: 状态空间维度（Acrobot为6）
    output_dim: 动作空间维度（Acrobot为3）
    """
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
        """输出为各个动作的概率分布"""
        return self.fc(x)


class ValueModel(nn.Module):
    """
    价值网络：输入状态，输出状态价值估计
    input_dim: 状态空间维度
    """
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
        """输出为状态价值估计 V(s)"""
        return self.fc(x)


class PPO:
    """
    PPO (Proximal Policy Optimization) 算法实现
    
    严格按照伪代码实现，包括：
    - GAE (Generalized Advantage Estimation) 优势估计
    - 小批量更新
    - 熵奖励项
    - KL 散度早停机制
    """
    def __init__(
        self,
        env,
        learning_rate=0.001,
        gamma=0.99,
        lamda=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
        target_kl=0.01,
        entropy_coef=0.01,
    ):
        self.env = env
        # 折扣因子 γ
        self.gamma = gamma
        # GAE 参数 λ，控制优势估计的偏差-方差权衡
        self.lamda = lamda
        # PPO 剪切参数 ε
        self.clip_eps = clip_eps
        # 策略更新次数 K
        self.epochs = epochs
        # 小批量大小 M
        self.batch_size = batch_size
        # 目标 KL 散度 δ
        self.target_kl = target_kl
        # 熵系数 c2
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.policy_model = PolicyModel(obs_dim, act_dim).to(self.device)
        self.value_model = ValueModel(obs_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def choose_action(self, state, deterministic=False):
        """
        选择动作
        deterministic=True: 取概率最大的动作（用于评估）
        deterministic=False: 从概率分布中采样（用于训练）
        """
        state = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            action_prob = self.policy_model(state)
        
        if deterministic:
            # 评估时：取概率最大的动作
            action = action_prob.argmax(dim=1).item()
        else:
            # 训练时：从概率分布中采样
            c = Categorical(action_prob)
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

    def compute_kl_divergence(self, old_probs, new_probs):
        """
        计算 KL 散度：KL[π_θ_old || π_θ] = Σ_a π_θ_old(a|s) log(π_θ_old(a|s) / π_θ(a|s))
        """
        old_probs = old_probs + 1e-8  # 避免 log(0)
        new_probs = new_probs + 1e-8
        kl = (old_probs * torch.log(old_probs / new_probs)).sum(dim=1).mean()
        return kl.item()

    def update(self, buffer):
        """
        按照伪代码严格实现的 PPO 更新方法
        
        输入：轨迹集 D_i = {(S_t, A_t, R_t, S_{t+1})}
        输出：更新后的策略参数 θ 和价值函数参数 W
        """
        # 步骤 2: 用策略 π_θ 收集轨迹集 D_i = {(S_t, A_t, R_t, S_{t+1})}
        states, actions, rewards, next_states, dones = zip(*buffer)
        T = len(states)  # 轨迹长度
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # 步骤 3: 使用当前价值网络计算 V_old(S_t) ← V(S_t; W)
        with torch.no_grad():
            v_old = self.value_model(states).squeeze()  # [T]
            v_old_next = self.value_model(next_states).squeeze()  # [T]
            # 对于终止状态，下一状态价值为 0
            v_old_next = v_old_next * (1 - dones)
            
            # 保存旧策略的动作概率，用于计算重要性采样比率和 KL 散度
            old_action_probs = self.policy_model(states)  # [T, action_dim]
            old_action_log_probs = torch.log(old_action_probs.gather(1, actions)).squeeze()  # [T]
        
        # 步骤 4: 初始化优势估计 A_t = 0
        # 步骤 5-8: 从后向前处理每个时间步，计算 GAE 优势估计
        advantages = torch.zeros(T, device=self.device)
        advantage = 0.0
        
        # 从后向前遍历：t = T-1, T-2, ..., 0
        for t in range(T - 1, -1, -1):
            # 步骤 6: 计算 TD 误差 δ_t = R_t + γ*V_old(S_{t+1}) - V_old(S_t)
            # 注意：对于终止状态（done=True），v_old_next[t] 已经被置为 0
            delta_t = rewards[t] + self.gamma * v_old_next[t] - v_old[t]
            
            # 步骤 7: 更新优势估计 A_t = δ_t + γ*λ*A_{t+1}
            advantage = delta_t + self.gamma * self.lamda * advantage
            advantages[t] = advantage
        
        # 步骤 9: θ_old ← θ（已通过 old_action_probs 保存）
        
        # 步骤 10: for 更新步 k = 1 to K do
        for k in range(self.epochs):
            # 步骤 11: 将 D_i 随机划分为小批量 {D'_1, D'_2, ...}
            indices = torch.randperm(T, device=self.device)
            
            for start_idx in range(0, T, self.batch_size):
                end_idx = min(start_idx + self.batch_size, T)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices].unsqueeze(1)  # [batch_size, 1]
                batch_v_old = v_old[batch_indices].unsqueeze(1)  # [batch_size, 1]
                batch_old_action_log_probs = old_action_log_probs[batch_indices].unsqueeze(1)  # [batch_size, 1]
                batch_old_action_probs = old_action_probs[batch_indices]  # [batch_size, action_dim]
                
                # 步骤 13: 计算重要性采样比率 r_t(θ) = π_θ(A_t|S_t) / π_θ_old(A_t|S_t)
                current_action_probs = self.policy_model(batch_states)  # [batch_size, action_dim]
                current_action_log_probs = torch.log(current_action_probs.gather(1, batch_actions))  # [batch_size, 1]
                ratio = torch.exp(current_action_log_probs - batch_old_action_log_probs)  # [batch_size, 1]
                
                # 步骤 14: 计算剪切目标 L_t^CLIP = min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)
                part1 = ratio * batch_advantages
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                part2 = clipped_ratio * batch_advantages
                clip_loss = -torch.min(part1, part2).mean()  # 取负号因为要最大化
                
                # 步骤 15: 计算目标回报 R_t ← A_t + V_old(s_t)
                target_returns = batch_advantages + batch_v_old  # [batch_size, 1]
                
                # 步骤 16: 计算价值损失 L_t^VF = (V(S_t; W) - R_t)^2
                current_values = self.value_model(batch_states)  # [batch_size, 1]
                value_loss = F.mse_loss(current_values, target_returns)
                
                # 步骤 17: 计算熵奖励 S_t = -Σ_a π_θ(a|S_t) log π_θ(a|S_t)
                dist = Categorical(current_action_probs)
                entropy = dist.entropy().mean()  # 熵（越大越好，鼓励探索）
                
                # 步骤 18: 更新策略 θ ← θ + β_θ * ∇_θ E[L_t^CLIP + c2*S_t]
                policy_loss = clip_loss - self.entropy_coef * entropy
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # 步骤 19: 更新价值函数 W ← W - β_W * ∇_W E[L_t^VF]
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
            
            # 步骤 21-23: 检查 KL 散度，如果 KL[π_θ_old || π_θ] > 1.5δ，则 break
            with torch.no_grad():
                current_probs = self.policy_model(states)
                kl_div = self.compute_kl_divergence(old_action_probs, current_probs)
                if kl_div > 1.5 * self.target_kl:
                    # 步骤 22: break（提前停止更新）
                    break


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