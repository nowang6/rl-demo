import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# ====== 超参数 ======
GAMMA = 0.99           # 折扣因子
EPS_CLIP = 0.2         # PPO 裁剪范围
LR = 1e-4              # 学习率
UPDATE_EPOCHS = 4      # 每次数据收集后的更新轮数
MAX_EPISODES = 500     # 最大训练回合数
HIDDEN_SIZE = 128      # 隐藏层大小
ENTROPY_COEF = 0.05    # 熵系数
VALUE_COEF = 0.5       # 价值损失系数
MAX_GRAD_NORM = 0.5    # 梯度裁剪阈值


class ActorCritic(nn.Module):
    """Actor-Critic 网络，共享特征提取层"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享特征提取层 - 简化但保持容量
        self.shared = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
        )
        # Actor 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic 输出状态价值
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def act(self, state):
        """选择动作并返回动作、对数概率、状态价值"""
        with torch.no_grad():
            action_probs, state_value = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, state_value

    def evaluate(self, state, action):
        """评估给定状态和动作的对数概率、状态价值、熵"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, state_value.squeeze(), entropy


def compute_returns(rewards, gamma=GAMMA):
    """计算折扣回报"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)


def train(max_episodes=MAX_EPISODES):
    """训练 PPO 智能体

    Args:
        max_episodes: 最大训练回合数
    """
    # 创建环境
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化模型和优化器
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("开始训练 PPO...")
    print(f"环境: CartPole-v1, 状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"超参数: gamma={GAMMA}, lr={LR}, eps_clip={EPS_CLIP}, update_epochs={UPDATE_EPOCHS}, entropy_coef={ENTROPY_COEF}, value_coef={VALUE_COEF}")
    print(f"训练回合数: {max_episodes}")

    for episode in range(max_episodes):
        # 收集一条轨迹的数据
        state, _ = env.reset()
        states, actions, rewards, log_probs, values = [], [], [], [], []

        done = False
        total_reward = 0

        while not done:
            # 将状态转换为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 选择动作
            action, log_prob, value = model.act(state_tensor)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # 存储数据
            states.append(state_tensor.squeeze(0))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.squeeze())

            # 更新状态
            state = next_state
            total_reward += reward

        # 计算回报和优势
        returns = compute_returns(rewards)
        values_tensor = torch.stack(values)
        advantages = returns - values_tensor.detach()

        # 优势标准化 (降低方差，稳定训练)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为张量
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(log_probs)

        # PPO 更新
        for _ in range(UPDATE_EPOCHS):
            # 评估当前策略
            new_log_probs, new_values, entropy = model.evaluate(states_tensor, actions_tensor)

            # 计算概率比
            ratios = torch.exp(new_log_probs - old_log_probs_tensor.detach())

            # 计算裁剪的代理目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            # 计算损失
            actor_loss = -torch.min(surr1, surr2).mean()
            # 价值损失 (MSE)
            critic_loss = 0.5 * nn.MSELoss()(new_values, returns)
            entropy_loss = -entropy.mean()  # 负熵，鼓励探索

            # 总损失
            loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

        # 输出训练进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d} | Reward: {total_reward:6.1f} | "
                  f"Loss: {loss.item():7.4f} | "
                  f"Avg Advantage: {advantages.mean().item():7.4f}")

    env.close()
    print("训练完成!")
    return model


def evaluate(model, num_episodes=10, render=False):
    """评估训练好的模型"""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = model(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"评估 Episode {episode + 1:3d} | Reward: {total_reward:6.1f}")

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n评估完成，共 {num_episodes} 回合")
    print(f"平均奖励: {avg_reward:.1f} ± {std_reward:.1f}")

    env.close()
    return episode_rewards


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="PPO 强化学习训练与评估")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="模式: train (训练) 或 eval (评估)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="训练回合数 (默认: 500)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="评估回合数 (默认: 10)")
    parser.add_argument("--render", action="store_true",
                        help="评估时渲染环境")
    parser.add_argument("--model_path", type=str, default="ppo_model.pth",
                        help="模型保存/加载路径 (默认: ppo_model.pth)")

    args = parser.parse_args()

    if args.mode == "train":
        model = train(max_episodes=args.episodes)
        # 保存模型
        torch.save(model.state_dict(), args.model_path)
        print(f"模型已保存到 {args.model_path}")

        # 评估训练好的模型
        print("\n开始评估训练好的模型...")
        evaluate(model, num_episodes=args.eval_episodes, render=args.render)

    elif args.mode == "eval":
        # 创建环境和模型
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        model = ActorCritic(state_dim, action_dim)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        print(f"加载模型 {args.model_path}")
        evaluate(model, num_episodes=args.eval_episodes, render=args.render)


if __name__ == "__main__":
    main()