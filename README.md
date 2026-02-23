# RL Demo: DQN 和 PPO 实现

一个使用 PyTorch 实现 Deep Q-Network (DQN) 和 Proximal Policy Optimization (PPO) 的强化学习演示项目。该项目使用 Gymnasium 环境，支持经典控制任务（如 CartPole）的训练和评估。

## 特性

- **完整的 DQN 实现**：包含 Q 网络、经验回放、目标网络等核心组件
- **模块化设计**：网络、缓冲区、智能体分离，易于扩展和维护
- **命令行接口**：支持训练和评估模式，丰富的参数配置
- **GPU 支持**：自动检测并使用 CUDA 设备（如果可用）
- **模型保存/加载**：训练进度可保存，支持中断后继续训练

## 项目结构

```
rl-demo/
├── dqn.py              # DQN 主实现文件
├── ppo.py              # PPO 主实现文件
├── main.py             # 项目入口点（示例代码）
├── pyproject.toml      # 项目配置和依赖
├── uv.lock             # 依赖锁文件
├── .python-version     # Python 版本
└── README.md           # 项目文档
```

## 安装

### 使用 uv（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd rl-demo

# 安装依赖（uv 会自动创建虚拟环境）
uv sync
```

### 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

## 依赖

- Python >= 3.12
- gymnasium >= 1.2.3
- torch >= 2.10.0
- numpy >= 2.4.2

## 快速开始

### 训练 DQN 智能体

```bash
# 使用默认参数训练 CartPole 环境（500 episodes）
python dqn.py --mode train --episodes 500

# 使用自定义参数训练
python dqn.py --mode train \
  --env "CartPole-v1" \
  --episodes 1000 \
  --lr 0.0005 \
  --gamma 0.99 \
  --batch_size 32 \
  --buffer_capacity 20000 \
  --save_path "my_dqn_model.pth"
```

### 评估训练好的模型

```bash
# 使用默认模型进行评估
python dqn.py --mode eval --load_path "dqn_model.pth"

# 不渲染环境进行评估（更快的批量评估）
python dqn.py --mode eval --load_path "my_dqn_model.pth" --no_render --num_eval_episodes 20
```

### 查看所有选项

```bash
python dqn.py --help
```

### 训练 PPO 智能体

```bash
# 使用默认参数训练 CartPole 环境（500 episodes）
python ppo.py --mode train --episodes 500

# 使用自定义参数训练
python ppo.py --mode train \
  --episodes 1000 \
  --eval_episodes 20 \
  --model_path "my_ppo_model.pth"
```

### 评估训练好的 PPO 模型

```bash
# 使用默认模型进行评估
python ppo.py --mode eval --model_path "ppo_model.pth"

# 渲染环境进行评估（可视化）
python ppo.py --mode eval --model_path "my_ppo_model.pth" --render

# 批量评估（不渲染，更快）
python ppo.py --mode eval --model_path "my_ppo_model.pth" --eval_episodes 20
```

### 查看 PPO 所有选项

```bash
python ppo.py --help
```

### 极简 PPO 教学版

我们还提供了一个极简教学版本，适合初学者理解和修改：

```bash
# 运行教学版 PPO (一个文件，200行以内)
python ppo_tutorial.py
```

教学版特点：
- **结构清晰**：一个文件，核心算法集中展示
- **一屏能看完**：200行以内，注释详细
- **新人能改**：模块化设计，易于实验
- **20分钟能跑起来**：CartPole环境快速收敛

## 命令行参数

### 训练模式 (`--mode train`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--env` | `CartPole-v1` | Gymnasium 环境名称 |
| `--episodes` | 1000 | 训练回合数 |
| `--lr` | 0.001 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--epsilon` | 1.0 | 初始探索率 |
| `--epsilon_decay` | 0.995 | 探索率衰减率 |
| `--epsilon_min` | 0.01 | 最小探索率 |
| `--target_update_freq` | 1000 | 目标网络更新频率（步数） |
| `--buffer_capacity` | 10000 | 经验回放缓冲区容量 |
| `--batch_size` | 64 | 训练批次大小 |
| `--eval_interval` | 10 | 训练中评估间隔（回合数） |
| `--save_path` | `dqn_model.pth` | 模型保存路径 |
| `--device` | `cuda` 或 `cpu` | 训练设备 |

### 评估模式 (`--mode eval`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--env` | `CartPole-v1` | Gymnasium 环境名称 |
| `--load_path` | `dqn_model.pth` | 模型加载路径 |
| `--num_eval_episodes` | 10 | 评估回合数 |
| `--no_render` | False | 禁用环境渲染（用于批量评估） |
| `--device` | `cuda` 或 `cpu` | 设备 |

### PPO 训练模式 (`--mode train`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 500 | 训练回合数 |
| `--model_path` | `ppo_model.pth` | 模型保存路径 |

### PPO 评估模式 (`--mode eval`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `ppo_model.pth` | 模型加载路径 |
| `--eval_episodes` | 10 | 评估回合数 |
| `--render` | False | 渲染环境（可视化） |

## DQN 实现细节

### 网络架构

```python
DQN(
  (fc1): Linear(in_features=state_dim, out_features=128)
  (fc2): Linear(in_features=128, out_features=128)
  (fc3): Linear(in_features=128, out_features=action_dim)
)
```

### 训练算法

1. **epsilon-greedy 策略**：平衡探索与利用
2. **经验回放**：随机采样打破数据相关性
3. **目标网络**：定期更新，稳定训练目标
4. **梯度裁剪**：防止梯度爆炸
5. **双网络架构**：当前网络 + 目标网络

### 超参数推荐

对于 CartPole-v1 环境：
- 学习率：`1e-3` 到 `1e-4`
- 折扣因子：`0.95` 到 `0.99`
- 批次大小：`32` 到 `128`
- 缓冲区容量：`10000` 到 `50000`

## 示例

### 使用示例代码

```python
# main.py 中的示例
from dqn import DQNAgent, train, evaluate

# 创建智能体
agent = DQNAgent(state_dim=4, action_dim=2)

# 训练
train(env_name="CartPole-v1", num_episodes=500)

# 评估
evaluate(env_name="CartPole-v1", model_path="dqn_model.pth")
```

### 训练输出示例

```
开始训练 DQN 于环境 CartPole-v1
设备: cuda
超参数: lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01
         target_update_freq=1000, buffer_capacity=10000, batch_size=64
------------------------------------------------------------
Episode   10 | Avg Reward:   18.30 | Avg Loss: 10.5243 | Epsilon: 0.9511 | Buffer:   640
Episode   20 | Avg Reward:   31.45 | Avg Loss:  8.7621 | Epsilon: 0.9048 | Buffer:  1280
Episode   30 | Avg Reward:   55.80 | Avg Loss:  6.1234 | Epsilon: 0.8607 | Buffer:  1920
...
训练完成。模型保存到 dqn_model.pth
```

## PPO 实现细节

### 网络架构

```python
ActorCritic(
  (shared): Sequential(
    (0): Linear(in_features=state_dim, out_features=64)
    (1): ReLU()
  )
  (actor): Sequential(
    (0): Linear(in_features=64, out_features=action_dim)
    (1): Softmax(dim=-1)
  )
  (critic): Linear(in_features=64, out_features=1)
)
```

### 训练算法

1. **Actor-Critic 架构**：共享特征提取层，分别输出动作概率和状态价值
2. **优势函数计算**：使用折扣回报减去状态价值估计
3. **PPO 裁剪目标**：限制策略更新幅度，保证训练稳定性
4. **熵正则化**：鼓励探索，防止策略过早收敛

### 核心公式

```python
# 概率比
ratios = torch.exp(new_log_probs - old_log_probs.detach())

# PPO 裁剪目标
surr1 = ratios * advantages
surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
actor_loss = -torch.min(surr1, surr2).mean()

# 优势函数
advantages = returns - values.detach()
```

### 超参数推荐

对于 CartPole-v1 环境：
- 学习率：`3e-4` (标准PPO学习率)
- 折扣因子：`0.99`
- PPO裁剪范围：`0.1` 到 `0.3`
- 更新轮数：`3` 到 `10`

### PPO 训练输出示例

```
开始训练 PPO...
环境: CartPole-v1, 状态维度: 4, 动作维度: 2
超参数: gamma=0.99, lr=0.0003, eps_clip=0.2, update_epochs=4
Episode   10 | Reward:    9.0 | Loss:  8.6204 | Avg Advantage:  4.3934
Episode   50 | Reward:   17.0 | Loss: 33.4976 | Avg Advantage:  7.8172
Episode  100 | Reward:   11.0 | Loss:  9.5820 | Avg Advantage:  2.8747
...
训练完成!
模型已保存到 ppo_model.pth
```

### 使用示例代码

```python
# 导入 PPO
from ppo import ActorCritic, train, evaluate

# 训练 PPO 智能体
train()

# 或使用命令行
# python ppo.py --mode train --episodes 500

# 评估训练好的模型
# python ppo.py --mode eval --model_path "ppo_model.pth" --render
```

## 支持的 Gymnasium 环境

- `CartPole-v1` (4维状态，2个动作)
- `MountainCar-v0` (2维状态，3个动作)
- `Acrobot-v1` (6维状态，3个动作)
- `Pendulum-v1` (3维状态，连续动作空间，需要调整)

## 扩展和定制

### 添加新环境

```python
# 只需更改 --env 参数
python dqn.py --mode train --env "MountainCar-v0"
```

### 修改网络架构

编辑 `dqn.py` 中的 `DQN` 类：

```python
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
```

### 实现 DQN 变体

项目结构支持轻松扩展：
- **Double DQN**：修改目标值计算
- **Dueling DQN**：修改网络架构
- **Prioritized Experience Replay**：修改 ReplayBuffer

### 修改 PPO 网络架构

编辑 `ppo.py` 中的 `ActorCritic` 类：

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 更深的网络
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
```

### 实现 PPO 变体

- **GAE (Generalized Advantage Estimation)**：修改优势函数计算
- **多环境并行采样**：使用 `gym.vector` 加速数据收集
- **Clipped Value Loss**：对价值损失也使用裁剪
- **自适应KL散度**：添加KL散度惩罚项
- **Prioritized Experience Replay**：修改 ReplayBuffer

## 故障排除

### 常见问题

1. **ImportError: No module named 'gymnasium'**
   ```
   uv sync  # 或 pip install gymnasium
   ```

2. **训练不收敛**
   - 尝试降低学习率 (`--lr 0.0001`)
   - 增加缓冲区容量 (`--buffer_capacity 50000`)
   - 调整探索率衰减 (`--epsilon_decay 0.999`)

3. **CUDA 内存不足**
   ```
   python dqn.py --device cpu  # 使用 CPU
   ```

4. **环境渲染问题**
   ```
   # 安装必要的渲染依赖
   pip install pygame pyglet
   ```

5. **PPO 训练不稳定**
   - 降低学习率 (`lr=1e-4`)
   - 减小PPO裁剪范围 (`eps_clip=0.1`)
   - 增加更新轮数 (`update_epochs=10`)
   - 调整熵系数 (`entropy_coef=0.02`)

6. **PPO 收敛速度慢**
   - 增加优势函数标准化
   - 使用GAE (Generalized Advantage Estimation)
   - 调整折扣因子 (`gamma=0.95` 到 `0.99`)

### 性能调优

- **GPU 加速**：确保 CUDA 可用时自动使用
- **批量大小**：根据 GPU 内存调整
- **经验回放**：适当容量避免过拟合

## 许可证

本项目使用 MIT 许可证。详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 参考

- [Mnih et al., 2015 - Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Schulman et al., 2017 - Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

---

*DQN 实现基于 DeepMind 的原始论文，PPO 实现基于 OpenAI 的论文，使用 PyTorch 和 Gymnasium 重新实现。*