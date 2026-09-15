# Arknights Zero — AlphaZero V0.1

日期：2026-09-15。真实关卡 0-1，支持池为黑角和玫兰莎。

**训练闭环已经实际运行，原有测试保持通过。共完成 13 iterations、65 个 self-play episodes、9,275 个样本、65 次 optimizer updates。最终神经网络 + PUCT 在固定评估和额外三个种子上均零漏怪通关；但是 65 局 self-play 全部失败，Value 很快饱和到 −1，评估成功率明显反复。当前证据支持“系统可用并能产生通关策略”，尚不支持“已经获得稳定、持续的自我改进”。**

本报告先完成 AlphaZero Core 验收。Desktop 工作此前按用户新指令暂停，未把未完成的 GUI 或打包计入本次成果。

## 1. Repository 结构

```text
arknights_sim/     既有 Simulator、GameData、State、Environment、Trace
agents/           既有 Random / Scripted / Plain UCT 回归基线
network/          State / Action encoders、PolicyValueNetwork、inference
mcts/             独立 neural PUCT、nodes、search statistics
training/         self-play、replay、trainer、events/control、config、stage sampler
configs/          alphazero_001.yaml，以及暂存的 Desktop 配置
scripts/          训练、评估、检查 policy、性能分析、曲线、回放命令
checkpoints/      发布的 latest.pt 与 best.pt
outputs/alphazero/ 全部实验、逐轮指标、逐局决策、图表与 policy 检查
research/alphazero/测试 XML、设备基准、profile 与运行日志
research/mechanics/仍未验证的四项底层机制
```

Core 不依赖 Qt；Simulator、MCTS 和训练均可通过命令行独立运行。

## 2. 新增文件

- `network/`：`state_encoder.py`、`action_encoder.py`、`policy_value_net.py`、`inference.py`、`__init__.py`、`README.md`。
- `mcts/`：`node.py`、`puct.py`、`search.py`、`tree_stats.py`、`__init__.py`、`README.md`。
- `training/`：`trainer.py`、`self_play.py`、`replay_buffer.py`、`events.py`、`stage_sampler.py`、`config.py`、`__init__.py`、`README.md`。
- `configs/alphazero_001.yaml`。
- `scripts/`：`train_alphazero.py`、`evaluate_alphazero.py`、`inspect_policy.py`、`benchmark_alphazero.py`、`profile_alphazero.py`、`plot_training.py`、`replay_alphazero_evaluation.py`。
- `tests/`：`test_network_alphazero.py`、`test_puct_alphazero.py`、`test_training_alphazero.py`、`test_alphazero_isolation.py`。

项目依赖/包发现配置相应扩展。没有重写原有 Simulator，也没有把 ScriptedAgent 或旧 UCT tactical rollout 导入 neural training 路径。

## 3. 网络架构

地图经过 96-channel 卷积和两个 residual blocks；干员/敌人分别使用共享 MLP，再做带 mask 的 mean/max pooling。全局信息和未来刷怪窗口经过小型 MLP。Fusion MLP 输出 256 维 state embedding。

每个合法 Action 经共享 Action MLP 编为 256 维，与 state query 做缩放点积并加 action bias，产生该动作的 logit。独立 Value MLP 输出一个 tanh 标量。没有 Transformer、固定巨型动作词表或游戏美术输入。

详见 [网络结构与输入说明](</Users/yihao/Desktop/arkzero/神经网络/network/README.md>)。

## 4. 参数数量

**1,097,314 个可训练参数**。Checkpoint 包含模型版本、各特征维度、channels、hidden size、未来窗口等 architecture metadata。

## 5. State encoding

| 输入 | 单状态 shape | 主要含义 |
|---|---|---|
| Map | `20 × H × W` | 地形、可部署类型、出入口、路线、单位占位/血量、阻挡、技能、坐标 |
| Operators | `O × 48` | 属性、部署/存活、方向、射程、费用、再部署、技能、攻击 cooldown |
| Enemies | `E × 32` | 属性、位置、路线进度、剩余距离、等待点、阻挡、攻击 cooldown |
| Global | `24` | 时间、DP、life、击杀/漏怪/已出生数量、队伍、地图和 episode 限制 |
| Future | `3 × 12` | 实际待出生队列在未来 5 / 10 / 20 秒内的聚合信息 |

默认实体上限为 8 operators / 64 active enemies，可配置；超过上限报错，不静默丢弃单位。地图、实体 padding 均有 mask，地图每层卷积也清除 padding，避免污染边缘格。

输入维持 Simulator 左下角原点，连续值使用固定物理参考尺度。未支持的官方职业不伪造，profession support flag 为 0。未部署干员的 HP/SP 特征表达下一次部署的初值，另有 deployed/alive/cooldown 标识；不是历史残血状态。完整状态仍保存在 GameState，神经网络观察是其确定性摘要，不等价于完整可逆序列化。

## 6. Action encoding

每个动态合法动作是 **80 维**：动作类型、对应干员 48 维属性、tile 坐标/类型/占位、方向、队伍序位和射程边界特征。输入来自统一不可变 `Action`，支持 DEPLOY / ACTIVATE_SKILL / RETREAT / WAIT。

动作按 Environment 返回顺序编码；训练样本保留同顺序的 canonical JSON identities，防止 policy 下标与动作错配。没有把某个坐标、干员 ID 或关卡时间线设成标准答案。

## 7. Policy 设计

Policy 是 **当前合法动作集合上的分布**。不存在全局 Action ID 0…N。非法动作从 Environment 入口即被排除；batch padding logits 为 `−inf`，softmax 概率为 0。计算交叉熵前将 padding 的 log-probability 清零，避免 `0 × −inf` 产生 NaN；测试确认 padding 无梯度。

## 8. Value 设计

Value 为 `[-1,1]` 的单玩家 outcome estimate。零漏怪且全部敌人击杀的完整通关标记 `+1`；其余 terminal episode 为 `−1`。同一局所有样本使用同一个最终 z，backup 不交替翻号。

Environment 的 decision/time limit terminal 也记失败，并单独记录 termination；用户中止的未结束 episode 被丢弃，不伪装成失败。Value 未经概率校准，不能解释为通关百分比。本次后期 Value 为 −1，但 PUCT 仍能通关，直接说明 Value 已失去有效区分度。

## 9. PUCT 实现

使用 `Q + c_puct × P × sqrt(N_parent) / (1 + N_child)`。节点保存稳定 state key、合法 actions、children、N/W/Q、terminal 与 prior，所有 backup 均从同一 Agent 视角累加。Terminal 使用环境精确值并绕过网络。

每次搜索按真实预算执行，统计 actual simulations / nodes / visits / depth / evaluator calls / cache hits / latency。只有唯一合法动作时返回该动作及 π=1，报告 **0 actual simulations**。没有复用 V0.2 搜索得到的成功攻略，也没有 rollout heuristic。Inference cache 只在单次搜索内有效，跨网络更新不复用旧预测。

## 10. Cold start

随机初始化；不使用专家轨迹、ScriptedAgent、已有 solution 或旧 tactical rollout。Self-play root 加 Dirichlet noise，alpha=0.3、epsilon=0.25；prior 同时可与 uniform 混合，全部合法动作保留。

| Iteration | Uniform mixture | Temperature |
|---|---:|---:|
| 0–2 | 0.50 | 1.00 |
| 3–5 | 0.30 | 1.00 |
| 6–10 | 0.15 | 0.75 |
| 11+ | 0.05 | 0.50 |

这些是 YAML 可改的初始实验配置，不是最优参数声明。评估始终关闭 noise、关闭 uniform mixture、temperature=0，避免随训练 schedule 改变比较标准。

## 11. Replay Buffer

CPU 上惰性增长的有界 deque，默认 capacity 50,000。样本包含 detached CPU state tensors、按序动作 JSON、对应 action features、π、terminal z。本实验最终 9,275 个样本，全部 `z=−1`。评估数据不进入 replay。

Replay 使用实际 batch 内的最大 action 数做 padding/mask。保存格式仅为 tensors + primitives，`torch.load(weights_only=True)`；原子替换文件，支持独立 save/load。不是一次预分配 50,000 局或把所有 tensor 常驻 MPS。

## 12. Training Loop

`AlphaZeroTrainer(config).train(callback=None, control=None)`：先评估随机网络；每轮用 accepted best 产生 self-play → 加入 replay → 采样 batch → AdamW 更新 candidate → 固定种子评估 → 比较 best → 保存 checkpoint 和日志。

使用 policy cross-entropy + value MSE，梯度裁剪 5，learning rate 0.001，batch 32，AdamW weight decay 0.0001。**`total_loss` 是真正优化的 policy + value；AdamW 使用 decoupled decay。`regularization_diagnostic` 另报 0.5λ‖θ‖²，不再次加入梯度，避免双重正则。**

Promotion 按成功率、return、少漏怪、击杀数比较。`promote_on_equal=true` 时同分明确记录 `tie_accepted`，不能称为进一步提升。Candidate optimizer 连续更新，self-play 使用当前 accepted best。

Qt-free `TrainingEvent` 为 immutable payload。`TrainingControl` 支持线程安全 pause/resume/stop，在决策与 optimizer step 边界响应；save/evaluate 请求在完成 iteration 边界执行。停止后保存已完成 episode 和 optimizer 进度，恢复时不重复加入数据。[完整训练接口与恢复语义](</Users/yihao/Desktop/arkzero/神经网络/training/README.md>)

## 13. Tests

| 范围 | Passed | Failed |
|---|---:|---:|
| 原 V0.1 Simulator | 56 | 0 |
| 原 V0.2 增量 | 82 | 0 |
| 新 Network | 11 | 0 |
| 新 PUCT | 31 | 0 |
| 新 Training | 12 | 0 |
| 新隔离/import guard | 1 | 0 |
| **合计** | **193** | **0** |

完整测试 19.167 秒，无 skipped/errors。[测试 XML](</Users/yihao/Desktop/arkzero/神经网络/research/alphazero/test_results.xml>)。覆盖输入形状/确定性/mask、合法动作/父子隔离、单玩家 backup、noise 与 temperature、动态 action loss、replay 保存、参数更新、stop/resume、checkpoint 后 CPU 推理和继续 optimizer 的精确一致。

另对 smoke 的 12 个评估 episode 在全新 Simulator 中重放：**12 / 12** 的动作时间、合法性、legal count、最终结果和成功率完全一致。原训练日志没有 trace/原始 state hash，所以该 audit **不声称与原运行 trace/hash 完全一致**。[回放审计](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/smoke/evaluation_replay_audit.json>)

## 14. CPU benchmark

实际机器为 **Apple M2 Pro、16 GiB、arm64**，Python 3.12.14、PyTorch 2.14.0、macOS 26.6.2。目标用户机器 M1 / 8 GB 尚未实测，不能把这里的结果当作 M1 测量值。正式训练采用 1 个 PyTorch CPU thread。

在真实 0-1、t=3 秒、49 个合法动作，热缓存且排除 GameData 加载后，CPU single-state inference median **0.775 ms**；batch 32 forward/backward/update median **40.089 ms**。设备基准中的人工 target 仅用于计时，不存入训练样本或 checkpoint。

## 15. MPS benchmark

MPS built/available 均为 true。同步并包括输出取回后，single-state inference median **2.515 ms**；batch 32 update median **12.438 ms**。计时点 current allocated 18.6 MB、driver allocated 84.6 MB，仅是基准测量时点，不是整轮训练峰值。

[CPU/MPS 完整样本与方法](</Users/yihao/Desktop/arkzero/神经网络/research/alphazero/device_benchmark.json>) · [实际硬件记录](</Users/yihao/Desktop/arkzero/神经网络/research/alphazero/runtime.json>)

## 16. 实际选择

**选择 CPU 跑完整训练**。当前逐节点单状态 PUCT 中 CPU 前向约比 MPS 快 3.25×；MPS 的 batch update 更快，但每 iteration 的更新次数少，不能抵消大量单状态推理开销。未来异步/批量叶节点评估后应重新测量。

## 17. Network inference time

CPU 纯 network 前向 0.775 ms，MPS 2.515 ms；State+Action encoding 的独立 median 为 **2.631 ms**，不能把网络前向时间当成完整 evaluator latency。计时统计见设备基准；profile 还分别记录 encoding、forward、clone 和 hash。

## 18. MCTS decision time

65 局 self-play 共 9,275 个决策，按决策加权的平均搜索时间 **8.606 ms**，累计 actual simulations **12,384**、nodes **21,052**，平均 **2.270 nodes/decision**。其中 **91.655%** 决策只有一个合法动作而跳过模拟，因此不能声称每个决策都执行了 16 simulations。

更能代表实际多动作搜索的最终 holdout 评估：平均 **70.054 ms/decision**、**16.340 nodes/decision**，forced fraction 约 1.736%。最终训练评估每局平均 decision latency 为 72.263 ms。数字对应不同策略产生的不同状态分布，不能直接当成速度回归。

## 19. Self-play episode time

65 局平均 **1.558 秒/局**，self-play 累计 101.273 秒。全部自然到达 `battle_end`，无 decision/time truncation；全部零通关。每轮计时之和 **299.000 秒**，包括该轮 self-play、更新、评估，不包括所有启动/加载/保存开销，不能冒充完整进程 wall time。

## 20. Training step time

真实 replay batch 32 的 65 次 optimizer update 平均 **44.099 ms/step**。包含 batch collation、前后向、裁剪、参数变化诊断等。每轮 `parameter_delta` 非零，测试也验证权重实际改变，未做空训练。

## 21. Smoke training

运行 **3 iterations × 5 episodes × 16 simulations × 5 optimizer updates**，每轮固定评估 3 局；为资源有界验证，CLI 覆盖了默认 YAML 的 32 simulations / 50 updates / 10 evaluations。

随机网络先评估 0/3，iteration 1 为 3/3，iteration 2 为 0/3，iteration 3 为 2/3。15 局 self-play 全部失败，生成 2,139 样本；iteration 1 best 被保留，未错误覆盖为退化的 iteration 3。smoke 原始数据和 checkpoint 全部保留。

## 22. 延长训练及全部曲线

从 smoke `latest.pt` 恢复 optimizer、replay、RNG 和迭代状态，继续 **10 个 iterations（4–13）**。沿用每轮 5 episodes / 16 simulations / 5 updates / 3 evaluations，未在恢复时悄悄更换算法配置。累计 65 self-play episodes、9,275 samples、65 updates。

| Iteration | Self-play 成功 | 固定评估成功 | Policy loss | Value loss | Total loss | Replay | 晋升 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 / 5 | 3 / 3 | 0.173718 | 0.503 | 0.677078 | 721 | 分数更好 |
| 2 | 0 / 5 | 0 / 3 | 0.191402 | 9.79e-09 | 0.191402 | 1426 | 保留旧 best |
| 3 | 0 / 5 | 2 / 3 | 0.141369 | 0 | 0.141369 | 2139 | 保留旧 best |
| 4 | 0 / 5 | 0 / 3 | 0.205908 | 0 | 0.205908 | 2858 | 保留旧 best |
| 5 | 0 / 5 | 3 / 3 | 0.146371 | 0 | 0.146371 | 3570 | 同分接受 |
| 6 | 0 / 5 | 3 / 3 | 0.248787 | 0 | 0.248787 | 4287 | 同分接受 |
| 7 | 0 / 5 | 3 / 3 | 0.055249 | 0 | 0.055249 | 5005 | 同分接受 |
| 8 | 0 / 5 | 1 / 3 | 0.115643 | 0 | 0.115643 | 5724 | 保留旧 best |
| 9 | 0 / 5 | 1 / 3 | 0.135519 | 0 | 0.135519 | 6432 | 保留旧 best |
| 10 | 0 / 5 | 3 / 3 | 0.213072 | 0 | 0.213072 | 7130 | 同分接受 |
| 11 | 0 / 5 | 3 / 3 | 0.197141 | 0 | 0.197141 | 7847 | 同分接受 |
| 12 | 0 / 5 | 1 / 3 | 0.236090 | 0 | 0.236090 | 8555 | 保留旧 best |
| 13 | 0 / 5 | 3 / 3 | 0.151086 | 0 | 0.151086 | 9275 | 同分接受 |


所有 self-play 成功列均为 0；所有固定评估和 self-play 均自然结束。Value loss 第 2 轮约 9.79e−9、第 3 轮起为 0，原因是 replay 标签全部负例且预测饱和，不是模型掌握了战斗。Policy loss 也明显反复。不能只展示 3/3 的轮次或把 loss 降低写成技能提升。

[全部逐轮原始指标](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/training_metrics.jsonl>) · [smoke 逐局轨迹](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/smoke/episodes.jsonl>) · [延长训练逐局轨迹](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/extended/episodes.jsonl>)

![全部成功率曲线](/Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/plots/success_rate.png)

![全部 loss 曲线](/Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/plots/loss.png)

![Replay 数量曲线](/Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/plots/replay_size.png)

## 23. 随机初始化模型表现

固定 seeds 80000–80002，随机网络 + PUCT：**0/3**，平均 0 kills / 11 leaks、146 decisions。额外 seeds 20001–20003：仍 **0/3**，平均 142.67 decisions。初始 Value 约 −0.0177，接近中性。

这些是当前两名干员、单一关卡和小样本结果。评估虽然关闭噪声和 temperature，搜索平分处理仍使用可复现种子；不能把三个种子当作大量独立环境泛化证据。

## 24. 训练后模型与基线

最终 candidate iteration 13 在固定评估 **3/3**，平均 11 kills / 0 leaks，因与 incumbent 同分接受为 best 13。额外 seeds 20001–20003 对所有五种 Agent 使用相同条件：

| Agent | 成功 | 平均击杀 | 平均漏怪 | 平均决策 | 平均每局秒 |
|---|---:|---:|---:|---:|---:|
| Untrained neural + PUCT | 0 / 3 | 0.00 | 11.00 | 142.67 | 1.531 |
| Current AlphaZero | 3 / 3 | 11.00 | 0.00 | 58.67 | 4.223 |
| RandomAgent | 0 / 3 | 0.00 | 11.00 | 143.00 | 0.155 |
| Plain UCT + tactical rollout | 3 / 3 | 11.00 | 0.00 | 43.00 | 2.198 |
| Scripted regression only | 3 / 3 | 11.00 | 0.00 | 13.67 | 0.099 |


[完整对比数据及每局结果](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/agent_comparison.json>)。

Plain UCT 基线使用旧版本的通用 tactical rollout 和搜索所得成功计划复用；Scripted 仅作历史回归。因此该表是实际行为对照，不是声称搜索计算预算或先验信息完全等价的公平算法竞赛。Neural self-play 不导入这些策略。

最终网络在 holdout 成功轨迹上的平均 Value 仍为 **−1**。当前提高的是“该 network + PUCT 组合在所测状态上的结果”，未证明 raw neural policy 独立通关能力、Value 可靠性或持续自我改进。三局 holdout 全成功也不足以建立高置信通关率。

## 25. Neural Policy 与 MCTS Policy 检查

检查 final iteration 13，真实 t=3 秒 / DP=13 / 49 legal actions，用 32 simulations、无 root noise。Value=−1.0，nonterminal。

| Action | Neural P | MCTS visit fraction | Visits |
|---|---:|---:|---:|
| WAIT | 0.045030 | 0.031250 | 1 |
| Melantha @ (5,2) DOWN | 0.025604 | 0.031250 | 1 |
| Melantha @ (6,3) DOWN | 0.025380 | 0.031250 | 1 |

最终选择 `(5,2) DOWN`。这里 32 个被访问动作各只有 1 次 visit，表中 π 是 raw `N / ΣN`，实际执行选择使用 temperature=0。**它证明 policy inspection 和 PUCT 重分配接口有效，但这一具体例子没有足够 Q/visit 区分证据，不能称为“搜索证实该动作更优”。**

[完整 final policy](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/final_policy.json>) · [初始网络检查](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/untrained_policy.json>) · [smoke best 检查](</Users/yihao/Desktop/arkzero/神经网络/outputs/alphazero/best_policy_initial_comparison.json>)

## 26. Checkpoint 文件

| 文件 | 状态 | 大小 |
|---|---|---:|
| [checkpoints/latest.pt](</Users/yihao/Desktop/arkzero/神经网络/checkpoints/latest.pt>) | iteration 13，可恢复完整训练 | 184,890,631 bytes（176.33 MiB） |
| [checkpoints/best.pt](</Users/yihao/Desktop/arkzero/神经网络/checkpoints/best.pt>) | accepted iteration 13，推理模型及匹配评估 | 4,403,377 bytes（4.20 MiB） |

两个发布文件与 extended 对应文件逐字节一致。全部 iteration 1–3 存于 smoke/checkpoints，4–13 存于 extended/checkpoints；原始实验未覆盖删除。latest/iteration 保存 model、best model、optimizer、config、CPU replay、RNG、pending progress、metrics；best 是明确的 inference-only，避免给 best 权重附上另一个 candidate 的 optimizer。

原子写入，安全 tensors/primitives 格式，加载使用 `weights_only=True`。已实测从 iteration 3 继续到 13；CPU 单元测试还验证中途 optimizer step 停止后恢复得到相同权重。跨设备/版本 bitwise reproducibility 未声称。

## 27. 当前 bottleneck 与机制边界

**学习瓶颈：正例缺失与 Value 饱和。** Self-play 经常部署后撤退，随后大量 forced WAIT；91.655% 决策的 action 只有一个。Uniform/noise 没有让这 65 局产生正奖励，Value 学会恒负，现有固定种子评估仍可能通关但更新不稳定。这比先增加模型规模更值得解决。

**计算瓶颈：特征/状态处理和模拟开销。** 一个带 cProfile 的独立 episode + update：environment step 累计 1.318 s、neural evaluation 1.057 s（含 encoding）、state hash 0.867 s、clone 0.699 s、state encode 0.598 s、action encode 0.185 s，而 model forward 0.340 s，update 0.046 s。累计值互相包含，不能相加；该 profile 同时有 smoke 任务运行且含 profiler 开销，只用于定位，不作为隔离速度基准。[Profile 说明与数据](</Users/yihao/Desktop/arkzero/神经网络/research/alphazero/profile/README.md>)。

下列规则仍全部 **ASSUMED**：连续转弯、阻挡半径、攻击前摇/伤害时机、同帧事件优先级。Neural 训练使用 Simulator 当前实现并不验证真实游戏。集中事件优先级保证确定性，但不代表已经还原真实客户端顺序。[机制目录](</Users/yihao/Desktop/arkzero/神经网络/research/mechanics/continuous_turning.md>)。

## 28. 下一阶段最值得优化的五件事

1. **解决正例采集和 Value 饱和**：保留纯 terminal 标签及所有失败记录，受控比较探索参数、搜索预算、温度和训练步数；先记录正例率、value 分布和训练梯度，避免只追 loss。
2. **单独审计决策分布**：统计部署后撤退、forced WAIT、有效多动作状态的样本比例。比较通用重加权/抽样实验，明确报告任何目标变化，不能注入标准攻略或静默删合法动作。
3. **扩大评估并建立消融**：增加未参与晋升的固定种子与初始条件，比较 raw neural policy、neural + PUCT、随机初始化 + PUCT，并保持相同搜索预算，验证改进来自哪里。
4. **依据 profile 优化开销**：先考虑 immutable 数据的编码复用、稳定 hash/clone 和批量叶节点推理；每次优化验证状态隔离和原 193 项测试。重新比较 CPU/MPS，先不引入多进程或语言重写。
5. **继续真实机制 golden 验证**：补连续转弯、阻挡半径、damage timing、同帧顺序的外部观测；只有证据充分才从 ASSUMED 升级，防止训练放大模拟偏差。
