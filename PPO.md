# PPO 计算公式说明

PPO（Proximal Policy Optimization）是一种强化学习算法，通过在每个更新中**限制策略更新的幅度**，在保证稳定性的同时保持较高的样本效率。其核心思想是：在优化策略时对单次更新施加约束，避免过大的策略变化导致性能崩溃。

---

## 1. PPO 目标函数（Clipped 目标）

PPO 的**剪切目标函数**（CLIP objective）为：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\; \text{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big) \hat{A}_t \right) \right]
$$

其中：

- **概率比** \( r_t(\theta) \)（当前策略与旧策略之比）：
  $$
  r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
  $$
  表示在状态 \( s_t \) 下采取动作 \( a_t \) 时，当前策略相对旧策略的概率比。

- **优势估计** \( \hat{A}_t \)：通常用 **GAE（广义优势估计）** 得到（见下节）。

- **裁剪超参数** \( \epsilon \)：控制单次更新允许的偏离幅度（常见取 \( 0.1 \sim 0.2 \)）。当 \( r_t(\theta) \) 超出 \( [1-\epsilon,\, 1+\epsilon] \) 时，通过 \(\text{clip}\) 限制更新，避免策略变化过大。

- **clip 函数**：
  $$
  \text{clip}(r,\, 1-\epsilon,\, 1+\epsilon) = \begin{cases}
  1+\epsilon & r > 1+\epsilon \\
  r & 1-\epsilon \le r \le 1+\epsilon \\
  1-\epsilon & r < 1-\epsilon
  \end{cases}
  $$

**直观含义**：  
- 当 \( \hat{A}_t > 0 \) 时，希望增大 \( \pi_\theta(a_t|s_t) \)，即希望 \( r_t(\theta) \) 变大，但被上限 \( 1+\epsilon \) 截断。  
- 当 \( \hat{A}_t < 0 \) 时，希望减小该动作概率，\( r_t(\theta) \) 被下限 \( 1-\epsilon \) 截断。  
这样在最大化 \( L^{\text{CLIP}} \) 的同时，自然限制了策略的更新步长。

---

## 2. 广义优势估计（GAE）

**TD 误差**（单步）：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**GAE 定义的** \( \hat{A}_t \)：
$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中：
- \( \gamma \)：折扣因子（discount factor），\( \in (0, 1] \)。
- \( \lambda \)：GAE 的权衡因子，\( \lambda=0 \) 退化为单步 TD，\( \lambda=1 \) 为整条轨迹的回报减去基线。

递推形式（便于实现）：
$$
\hat{A}_t = \delta_t + (\gamma\lambda) \hat{A}_{t+1}
$$
从轨迹末端向前递推即可。

---

## 3. 完整 PPO 损失（带价值与熵）

实际实现中，总损失通常由三部分组成：

$$
L(\theta) = \mathbb{E}_t \left[ L^{\text{CLIP}}_t(\theta) - c_1 L^{\text{VF}}_t(\theta) + c_2 S\big[\pi_\theta(\cdot|s_t)\big] \right]
$$

- **策略项** \( L^{\text{CLIP}}_t(\theta) \)：即上面的剪切目标（对单步或 mini-batch 取平均）。
- **价值损失**（如 MSE）：
  $$
  L^{\text{VF}}_t(\theta) = \big( V_\theta(s_t) - V_t^{\text{targ}} \big)^2
  $$
  其中 \( V_t^{\text{targ}} \) 常用回报或 GAE 构造的目标（如 \( \hat{A}_t + V(s_t) \)）。
- **熵项** \( S[\pi_\theta(\cdot|s_t)] \)：策略在 \( s_t \) 下的熵，用于鼓励探索；\( c_1,\, c_2 \) 为系数（如 0.5 和 0.01）。

对 \( L(\theta) \) 做**梯度上升**（或对 \(-L\) 做梯度下降）即可更新策略与价值网络。

---

## 4. 小结

| 符号 | 含义 |
|------|------|
| \( \pi_\theta \) | 当前策略（带参数 \( \theta \)） |
| \( \pi_{\theta_{\text{old}}} \) | 旧策略（更新前或 behavior 策略） |
| \( r_t(\theta) \) | 概率比 \( \pi_\theta(a_t\|s_t) / \pi_{\theta_{\text{old}}}(a_t\|s_t) \) |
| \( \hat{A}_t \) | 优势估计（常用 GAE） |
| \( \epsilon \) | clip 范围 \( [1-\epsilon,\, 1+\epsilon] \) |
| \( \gamma \) | 折扣因子 |
| \( \lambda \) | GAE 的 \( \lambda \) |

PPO 通过 **min(clip(...), r_t·Â_t)** 的形式，在最大化期望优势的同时，把概率比限制在 \( 1 \pm \epsilon \) 内，从而避免大幅度策略更新，提高训练稳定性。若需对照代码，可结合项目中的 `ppo.py` 查看 \( L^{\text{CLIP}} \)、GAE 和总损失 \( L(\theta) \) 的实现。
