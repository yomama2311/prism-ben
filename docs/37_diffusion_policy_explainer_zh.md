# 37 — Diffusion Policy 完全讲解（从扩散模型到我们的 rebuttal 发现）

> 写给 Yuhai 的学习文档（2026-08-06，rebuttal 周）。目标：把 DP 的原理、我们仓库里的实现、
> 以及这次 rebuttal 中围绕 DP 的全部发现串成一条线。读完应该能在 post-rebuttal 讨论里
> 独立回答任何 DP 相关的问题。原文：Chi et al., "Diffusion Policy: Visuomotor Policy
> Learning via Action Diffusion", RSS 2023 (arXiv 2303.04137)。

---

## 1. DP 要解决什么问题：模仿学习的多模态困境

Behavior cloning（BC）的朴素做法是训练一个网络 `f(obs) → action`，用 MSE 拟合专家动作。
这在**专家行为唯一**时没问题，但真实演示往往是**多模态**的：

- PushT 里，从 T 块下方把它推进目标区，专家有时从**左边**绕、有时从**右边**绕。
  两条路都对。
- MSE 回归会输出两个模式的**平均值**——直直地撞向 T 块中间，两边都不是。

原文 Fig. 4 用这个场景对比了几类旧方案的失败方式：

| 方案                     | 失败方式                                          |
| ---------------------- | --------------------------------------------- |
| MSE 回归                 | 模式平均（撞中间）                                     |
| GMM 头（LSTM-GMM/BC-RNN） | 要预先猜模式个数；训练不稳                                 |
| 离散化（BET）               | 动作空间高维时离散化爆炸；序列内模式漂移（左一步右一步，抖动）               |
| 隐式能量模型（IBC/EBM）        | 理论漂亮，但 InfoNCE 训练要估配分函数 Z(o,a)，负样本不准 → 训练极不稳定 |

DP 的回答：**用扩散模型表示 `p(动作序列 | 观测)`**。扩散模型天生擅长表达高维多模态分布
（图像生成早已证明），而且训练目标是一个普通的 MSE（见 §2），完全绕开配分函数。
每次采样从不同的高斯噪声出发、收敛到不同的模式——**一次 rollout 内自然承诺单一模式**，
不会左右横跳。

## 2. 背景：扩散模型 60 秒版

**前向过程**：把干净数据 `x⁰`（这里是动作序列）逐步加高斯噪声，K 步之后变成纯噪声 `xᴷ`。

**反向过程（学习目标）**：训练网络 `ε_θ(x^k, k)` 预测"第 k 步的样本里掺了什么噪声"。
训练损失就是：

```
L = MSE( ε,  ε_θ( x⁰ + ε·(噪声调度), k ) )        # ε 是真实加入的噪声
```

**采样**：从纯高斯噪声出发，反复调用 `ε_θ` 把噪声一点点减掉，K 步后得到一个干净样本。
每一步的转移是 `x^{k-1} = α(x^k − γ·ε_θ(x^k,k)) + N(0, σ²I)` —— 注意每步转移是
**高斯的**（这是 rebuttal 里"PoG applies to any Gaussian sampler"那句话的数学基础）。

**DDPM vs DDIM**：训练时用 K=100 步的完整调度（DDPM）；推理时不必走满 100 步，
DDIM 允许跳步——我们的实现推理只走 **10 步**，速度 ×10，质量基本不掉。
原文 §IV.D：DDIM 10 步让真机推理延迟压到 0.1s（3080 GPU）。

**为什么比 EBM 稳**（原文 §IV.D 的核心洞察）：ε_θ 学的是 score function
`∇ log p(a|o)`，而 score 与配分函数 Z(o,θ) **无关**（∇ log Z = 0）——
IBC 死于估 Z，DP 从数学上绕开了 Z。

## 3. DP 的三个关键设计（原文 §II.C / §III）

### 3.1 输出动作**序列**，不是单步动作

一次预测 T_p 步动作（closed-loop action-sequence prediction）。好处：
- **时间一致性**：整段序列出自同一个模式，不会步间换模式；
- **对 idle actions 鲁棒**：人类演示里常有停顿（等待、调整），单步 BC 会学会"卡住"，
  序列预测把停顿放进上下文里消化掉。

### 3.2 条件是观测特征：p(A_t | O_t)

观测（最近 T_o 帧图像+本体状态）经视觉编码器变成特征 O_t，用 **FiLM**
（每层卷积做 feature-wise 仿射调制）注入去噪网络。去噪的对象只有动作序列——
不预测未来观测（这是它和 Diffuser/世界模型路线的分水岭：**DP 不建模世界，只建模行为**）。

### 3.3 Receding horizon：T_o / T_p / T_a

每个时刻：吃 T_o 帧观测 → 预测 T_p 步动作 → **只执行前 T_a 步** → 重新观测、重新预测。
原文默认 T_p=16, T_a=8, T_o=2。Fig. 6 消融：T_a 太短 → 反应快但时间一致性差；
太长 → 一致但迟钝；**T_a=8 是他们任务上的峰值**。

⚠️ 这个消融有一个隐含前提，见 §6——它是我们整个 rebuttal 里最重要的发现的来源。

### 3.4 架构两个版本

- **CNN 版**（我们用的思路）：Conv1D U-Net 沿时间轴处理动作序列，FiLM 注入观测特征
  和扩散步 k 的 embedding。开箱即用，超参不敏感；缺点是低通倾向（动作变化剧烈时过平滑）。
- **Transformer 版**：minGPT 式 decoder，动作 token 做 causal attention，观测特征做
  cross-attention。上限更高但更难调。
- 视觉编码器：ResNet18（不预训练），全局池化 → **spatial softmax**（保空间信息），
  BatchNorm → **GroupNorm**（配合 EMA 稳定），端到端联合训练。评测用 **EMA 权重**。

## 4. 我们仓库的实现（`dp_baseline/`，docs/23）

自写 ~700 行（**没有用** diffusers 库和原版仓库——原版自带数据格式/评测协议/60M ResNet
一整套，混进来全是混杂变量）。关键差异和参数：

| 项 | 原版 DP | 我们的 dp_baseline |
|---|---|---|
| 任务条件 | 仅观测（**没有 goal 输入**，任务烙在权重里） | **观测 + goal 图像**（concat 特征做 FiLM 条件）|
| 视觉编码器 | ResNet18 + spatial softmax | 小 CNN（224×224 → 256 维）|
| 预测窗口 T_p | 16 | **25**（对齐 planner 的 H=5 × block=5）|
| 执行步数 T_a | 8 | 8（沿用原版惯例 → 见 §6 的坑）|
| 参数量 | ~60M+（含 ResNet） | **19.3M** |
| 归一化 | 动作 min-max 到 [-1,1] | 同 |
| 调度 | DDPM 100 训练 / DDIM 推理 | DDPM 100 / **DDIM 10** |
| 训练量 | — | 100k 步、batch 128、1.8M 窗口、EMA 0.999、~4.2h(PushT) |

goal 对齐修复（docs/23 §23.4）：最初训练时 goal 取 episode 末帧，而 eval 的 goal 在
start+25 —— 修成 start+25 后 PushT 从 23.3 → 40.7（+17.4pp）。**这是第一次**
goal-偏移分布错配教训（训练侧）。第二次在评测侧，见 §6。

关键文件：`dp_baseline/{model,scheduler,dataset,train,policy}.py`、
`eval_diffusion_policy.py`（评测入口，支持 `+dp.n_action_steps=25`、
`+dp.eval_ep_filter=dp_val`）。

### 4.1 DP 是怎么"知道"目标的？

**原版 DP：不知道。** 输入只有观测，没有 goal 通道。所有演示都推向同一个固定目标区，
"追求目标"被蒸馏进权重——目标是出厂设置，换目标只能重训。

**我们的 goal-conditioned 变体：目标 = 一张"未来照片"。**

1. 评测框架把演示第 start+25 步的画面作为 `info_dict["goal"]` 传入（整条 episode 不变）；
2. goal 图与观测图各自过同一个 CNN 编码器 → `cond = concat(obs_feat, goal_feat)`；
3. `cond` 经 FiLM 调制去噪 UNet 的每一层——目标图片在去噪的每一步都在"掰"内部特征，
   把去噪引向"能连接这两张图的那段动作片段"。

**关键理解**：DP "知道"的不是目标本身，而是**图像对 → 动作片段**的统计关联。它对
"到达那里"没有概念，所以目标间隔一漂移（25→17步）、或目标落在训练联合分布外
（末帧 vs 25步），关联就断了——§6 的病理和 docs/23 的 goal 修复共享这一个根源。
对照：WM 规划同样拿这张图，但编码后**当靶子打分**（度量对象），不当条件输入——
这就是 §7 表里 "goal 进入方式" 那一行的全部含义。

## 5. DP 为什么在我们的 benchmark 上（修正后）这么强

修正后 93.3 (PushT) / 99.3 (Cube)。三个原因叠加：

1. **考题就是训练题型**：eval 从演示分布取起点、goal 是同一条演示 25 步后的状态——
   这字面上就是 DP 的训练目标 `(obs_t, goal_{t+25}) → a[t:t+25]`；
2. **数据密**：18.7k 条 PushT 演示把状态空间铺满。注意这**不是背题**——DP 在它从未
   见过的 2% episode 上照样 88.7/100（我们实测），是分布内泛化；
3. **任务准静态**：25 步开环执行在无扰动仿真里零代价。

这也是 rebuttal §2 "the protocol favors behavior cloning" 的完整含义。

## 6. 本次最大的发现：goal conditioning × receding horizon 的病理

### 6.1 原作者早就预警过

原文 §III-A 第三条修改（原话）：

> "we removed inpainting-based goal state conditioning **due to incompatibility with
> our framework utilizing a receding prediction horizon**. However, goal conditioning
> is still possible with the same FiLM conditioning method used for observations."

他们删掉了 goal conditioning，只留了一句"理论上可以用 FiLM 做"——**没有实验**。
我们的 benchmark（沿 LeWM 协议）恰好把"FiLM goal conditioning + 固定偏移 goal +
receding horizon"三件组合在了一起。

### 6.2 病理时间线（T_a=8 时）

DP 训练中 (obs, goal) 的时间间隔**恒等于 25 步**。评测中 goal 固定在 t=25：

| 重规划时刻 | 条件 (obs, goal) 间隔 | 训练时见过吗 |
|---|---|---|
| t=0 | 25 步 | ✅ |
| t=8 | **17 步** | ❌ |
| t=16 | **9 步** | ❌ |
| t=24 | **1 步** | ❌ |

模型没有时钟输入，"目标还有多远"是从图像对的视觉差推断的——而训练集里这个差
恒定对应 25 步节奏。间隔 17/9/1 的图像对落在训练流形外 → 输出退化 → 41%。

### 6.3 修复与证据链

把 T_a 设为 25（= goal 偏移 = 我们 planner 的重规划周期），条件分布回到训练支撑集：

- 同一 checkpoint、同一脚本、只改这个数：PushT **40.7 → 93.3**、Cube **76.7 → 99.3**；
- best-of-1 对照（我们的 `eval_dp_prior.py` 路径）91.3/99.3 复核了数字；
- 原版 benchmark 免疫的原因：它**根本没有 goal 输入**，无所谓间隔漂移。

### 6.4 一般教训

1. **BC 的"目标关系"能力是数据构造喂出来的**：如果训练时把 goal 偏移随机化（5~40 步），
   DP 就能学会看图估距离，T_a=8 也不会崩——这是 camera-ready 的"满血 DP"修法。
2. **WM 规划在这一类病上结构免疫**：goal 不是 WM 的条件输入，只是打分的靶子——
   cost = ‖预测 embedding − goal embedding‖ 对任何目标图像、任何距离都有定义。
   （注意精确表述：免疫的是**打分层**；我们的 prior head 同样有 goal 输入、同样会被
   偏移打伤——BC-only 行 31% 就是证据。PRISM 活下来靠的是 planner 重打分。）
3. **超参的适用前提要跟着任务结构重新检查**：T_a=8 在原版是最优，在 goal-conditioned
   协议里是毒药。我们给自己的方法扫了 σ_scale 九个值、却没给 DP 扫 T_a——这是
   rebuttal §2 承认的调参不对称。

## 7. DP vs PRISM：两种范式一张表

| | DP（学行为） | PRISM（学先验 + WM 规划） |
|---|---|---|
| 回答的问题 | "专家在这里做过什么" | "什么动作能到达**这个**目标（据我学的物理）" |
| goal 进入方式 | 条件输入（须在训练联合分布内） | 打分靶子（任意目标图像可用） |
| 输出错误的后果 | 直接执行 | 先验只是提议，候选全部过 WM cost 复核 |
| 数据 50% 被污染 | **93.3→55.3，无预警**（val loss 0.021→0.032 几乎不动） | 85.4→66.7，**σ̄ 0.53→3.08 自动报警**，ε=100% 收敛到 vanilla 不跌破 |
| goal 偏移 75-175 步 | −17.4pp（更小偏移就崩） | 82-89% 不动（planner 重锚定） |
| 参数 / 先验延迟 | 19.3M / ~70ms(16样本) | <1M / 0.048ms |
| 系统级推理开销 | **更便宜**（每25步一次DDIM） | 更贵（K×30 次 WM rollout）——买的是验证 |
| 在饱和 benchmark 上 | 93-99（主场） | 85.4/83.0 |
| 强项场景 | 任务固定、数据充足干净、分布内 | 预算小、目标灵活、数据质量未知、需要运行时验证 |

## 8. 全部 DP 相关实测数字备忘（本仓库）

| 实验 | PushT | Cube | 出处 |
|---|---|---|---|
| DP paper 行（T_a=8） | 41±10 | 77±5 | docs/23, 论文 Tab.1 |
| DP 修正（T_a=25） | 93.3±5.0 | 99.3±1.2 | logs/rebuttal_dp_cadence.tsv |
| DP 未见过的 2% episodes | 88.7±6.1 | 100±0 | logs/rebuttal_dp_valeps.tsv |
| DP best-of-16 + WM verify | 95.3±4.2 | 100±0 | logs/rebuttal_hybrid_results.tsv |
| DP→init (PiJEPA-style) | 94.7±1.2 | 98.7±1.2 | 同上 |
| DP→PoG | 96.0±0.0 | 98.0±2.0 | 同上 |
| best-of-1 对照 | 91.3±6.4 | 99.3±1.2 | 同上 |
| DP@ε=50% 污染数据 | 55.3±6.4 | — | logs/rebuttal_dp_corrupt.tsv |
| 训练预算 | 19.3M 参数 / 100k 步 / 1.8M 窗口 / 98% episodes | | docs/23 |
| 先验采样延迟（16样本×DDIM-10, 50 envs, 空闲GPU） | 69.5ms | 69.5ms | logs/rebuttal_timing.log |

## 9. 延伸阅读

- 原文精读顺序：§II.C（三个设计）→ §III-A（那句 goal 预警）→ §IV（多模态/稳定性直觉）
  → Fig. 6（T_a 消融）
- 本仓库：docs/21（goal 错配初诊）、docs/23（DP baseline 全记录）、
  rebuttal §2（诊断的最终定稿表述）
- 对照阅读：docs/03（PoG 机制）——理解"两种范式"那张表的 PRISM 侧
