# RL Demo: DQN 与 PPO 实现

使用 PyTorch 实现 **Deep Q-Network (DQN)** 和 **Proximal Policy Optimization (PPO)** 的强化学习演示项目，基于 Gymnasium 环境，支持经典控制任务（如 CartPole、Acrobot）的训练与评估。

---

## 1. 特性

- **完整算法实现**：DQN（Q 网络、经验回放、目标网络）与 PPO（策略-价值网络、GAE、clip 目标）
- **模块化设计**：网络、缓冲区、智能体分离，便于扩展与维护
- **命令行接口**：训练/评估模式、丰富参数，`--help` 查看全部选项
- **GPU 支持**：自动检测并使用 CUDA（如可用）
- **模型保存与加载**：支持断点续训与离线评估

---

## 2. 安装与依赖

### 2.1 环境要求

- Python >= 3.12
- gymnasium >= 1.2.3
- torch >= 2.10.0
- numpy >= 2.4.2

### 2.2 使用 uv（推荐）

```bash
git clone <repository-url>
cd rl-demo
uv sync
```

### 2.3 使用 pip

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
pip install -e .
```

---

## 3. 项目结构

```
rl-demo/
├── dqn.py              # DQN 训练与评估入口
├── ppo.py              # PPO 训练与评估入口
├── main.py             # 项目入口示例
├── pyproject.toml      # 项目配置与依赖
├── uv.lock             # 依赖锁文件
├── .python-version     # Python 版本
└── README.md           # 本说明
```

---

## 4. 使用说明

### 4.1 DQN（CartPole 等）

**训练**

```bash
# 默认：CartPole-v1，500 episodes
python dqn.py --mode train --episodes 500

# 自定义环境与超参
python dqn.py --mode train \
  --env "CartPole-v1" \
  --episodes 1000 \
  --lr 0.0005 \
  --gamma 0.99 \
  --batch_size 32 \
  --buffer_capacity 20000 \
  --save_path "my_dqn_model.pth"
```

**评估**

```bash
# 加载模型并评估（默认渲染）
python dqn.py --mode eval --load_path "dqn_model.pth"

# 不渲染、多局评估
python dqn.py --mode eval --load_path "my_dqn_model.pth" --no_render --num_eval_episodes 20
```

**查看选项**

```bash
python dqn.py --help
```

---

### 4.2 PPO（Acrobot 等）

**训练**

```bash
# 激活虚拟环境（若未激活）
source .venv/bin/activate   # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 默认：Acrobot-v1，300 episodes，模型保存为 ppo_acrobot.pt
python ppo.py

# 指定回合数
python ppo.py --episodes 500
```

训练结束后会：保存模型到 `ppo_acrobot.pt`、弹出回报曲线图、并自动演示一轮动画。

**训练流程简述**

1. 创建 `Acrobot-v1` 环境（`render_mode='rgb_array'` 用于可视化）
2. 构建 PPO 智能体（策略网络 + 价值网络 + 优化器）
3. 按 episode 循环：收集轨迹 → GAE + PPO clip 更新 → 记录回报，约每 30 个 episode 打印一次
4. 绘制回报曲线、保存 `ppo_acrobot.pt`、用当前策略确定性跑一轮并渲染演示

**评估**

```bash
# 仅数值评估：加载默认 ppo_acrobot.pt，跑 10 局，输出平均/最小/最大回报
python ppo.py --eval

# 指定模型与局数
python ppo.py --eval --checkpoint ppo_acrobot.pt --episodes 20

# 评估后再渲染一局（弹窗动画）
python ppo.py --eval --render
```

**查看选项**

```bash
python ppo.py --help
```

**PPO 常用参数**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| （无） | 训练模式 | - |
| `--eval` | 仅评估，不训练 | - |
| `--episodes` | 训练回合数或评估局数 | 训练 300，评估 10 |
| `--checkpoint` | 模型文件路径 | `ppo_acrobot.pt` |
| `--render` | 评估后是否再渲染一局 | False |

---

## 5. 常见问题

- **PPO 报错“未找到模型”**：先执行一次 `python ppo.py` 完成训练，再使用 `python ppo.py --eval`。
- **想多测几局**：评估时加 `--episodes 20`（或更大）。
- **只想看动画效果**：使用 `python ppo.py --eval --render`。
- **无图形界面**：评估时不加 `--render`，仅用 `--eval` 即可得到数值统计；训练时的回报曲线若无法弹窗，可后续从保存的数据自行绘图。
- **自定义 PPO 保存路径**：在 `ppo.py` 的 `main()` 中修改保存路径，或评估时通过 `--checkpoint your_path.pt` 指定。
