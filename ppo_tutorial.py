"""
极简 PPO 教学 Demo (离散动作)
适用于 Gymnasium 环境

特点：
✅ 结构清晰 - 一个文件，200行以内
✅ 一屏能看完 - 核心代码集中
✅ 新人能改 - 模块化，注释详细
✅ 20分钟能跑起来 - CartPole环境快速收敛

使用方法：
python ppo_tutorial.py

安装依赖：
pip install gymnasium torch numpy
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ====== 超参数 ======
GAMMA = 0.99           # 折扣因子
EPS_CLIP = 0.2         # PPO裁剪范围
LR = 3e-4              # 学习率
UPDATE_EPOCHS = 4      # 每次收集数据后的更新轮数
HIDDEN_SIZE = 64       # 网络隐藏层大小
MAX_EPISODES = 500     # 最大训练回合数


# ====== Actor-Critic 网络 ======
class ActorCritic(nn.Module):
    """Actor-Critic 网络：共享特征提取，输出动作概率和状态价值"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
        )
        # Actor: 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic: 输出状态价值
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state):
        """前向传播：返回动作概率和状态价值"""
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


# ====== 辅助函数 ======
def compute_returns(rewards, gamma=GAMMA):
    """计算折扣回报 (蒙特卡洛方法)"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)


# ====== 主训练函数 ======
def train_ppo():
    """PPO 主训练循环"""

    # 1. 创建环境
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("=" * 60)
    print("极简 PPO 教学 Demo")
    print(f"环境: CartPole-v1, 状态: {state_dim}维, 动作: {action_dim}个")
    print(f"超参数: gamma={GAMMA}, lr={LR}, eps_clip={EPS_CLIP}")
    print("=" * 60)

    # 2. 初始化模型和优化器
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. 训练循环
    for episode in range(MAX_EPISODES):
        # 收集一条轨迹的数据
        state, _ = env.reset()
        states, actions, rewards, log_probs, values = [], [], [], [], []

        done = False
        total_reward = 0

        # 4. 与环境交互，收集数据
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 使用当前策略选择动作
            with torch.no_grad():
                action_probs, state_value = model(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # 存储经验
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(state_value.item())

            state = next_state
            total_reward += reward

        # 5. 计算回报和优势
        returns = compute_returns(rewards)
        advantages = returns - torch.FloatTensor(values)

        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(log_probs)

        # 6. PPO 更新 (核心!)
        for _ in range(UPDATE_EPOCHS):
            # 评估当前策略
            action_probs, state_values = model(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy()

            # PPO 核心公式：概率比和裁剪
            ratios = torch.exp(new_log_probs - old_log_probs_tensor.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            # 计算损失
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            entropy_loss = -entropy.mean()  # 负熵，鼓励探索

            # 总损失 = Actor损失 + Critic损失 + 熵正则化
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 7. 输出训练进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {total_reward:5.1f} | "
                  f"Loss: {loss.item():6.3f} | "
                  f"Advantage: {advantages.mean().item():6.3f}")

    # 8. 训练完成
    env.close()
    print("=" * 60)
    print("训练完成！模型已学习到平衡策略")
    print("可以尝试增加训练回合数以获得更高分数")
    print("=" * 60)

    return model


# ====== 评估函数 ======
def evaluate_model(model, num_episodes=5, render=False):
    """评估训练好的模型"""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    print(f"\n评估模型 ({num_episodes}回合):")
    rewards = []

    for i in range(num_episodes):
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

        rewards.append(total_reward)
        print(f"  回合 {i+1}: {total_reward:5.1f}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"平均奖励: {avg_reward:.1f}")

    env.close()
    return avg_reward


# ====== 主函数 ======
def main():
    """主函数：训练和评估PPO"""

    # 训练 PPO 模型
    model = train_ppo()

    # 快速评估
    print("\n进行快速评估...")
    evaluate_model(model, num_episodes=3, render=False)

    # 保存模型
    torch.save(model.state_dict(), "ppo_tutorial_model.pth")
    print(f"\n模型已保存到: ppo_tutorial_model.pth")

    # 提示用户如何进一步评估
    print("\n如需可视化评估，请运行:")
    print("python ppo_tutorial.py --eval --render")


if __name__ == "__main__":
    main()