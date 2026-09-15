# Arknights Zero — Simulator V0.2 / MCTS-Ready Environment

日期：2026-09-15。状态：**按用户要求，完成当前参数实验、保存数据后暂停。** 核心环境、MCTS 与回归已经完成；完整性能基准尚未运行，不将 V0.2 最终交付标记为全部完成。 在保留 V0.1 的基础上，Agent 已能完整控制真实 GameData 0-1；单人 UCT MCTS 以 **16 次模拟、seed 12345** 自行找到 **11/11 击杀、0 漏怪、20 耐久** 的成功方案，并在全新环境中逐决策、逐状态、完整战斗 trace 重放一致。

这里的“真实 0-1”指固定的官方 GameData 关卡输入，通关是在当前 Simulator 内完成。连续转弯、阻挡半径、攻击前摇、同帧优先级仍全部为 **ASSUMED**。MCTS 与 replay 成功不构成实机精度验证。

## 1. 保留与修改范围

首先检查了原有 State、Stage、单位、事件队列、战斗、路线、阻挡、伤害、部署和 trace 架构，并运行原有完整测试：**56 passed / 0 failed**，原始报告保留在 `research/test_results.xml`。没有删除或重建模拟器，没有修改真实数据或原有测试。

| 文件 | 修改内容 |
| --- | --- |
| `arknights_sim/core/simulator.py` | 增加可选 settled-boundary observer；集中公开阶段顺序；缓存不可变路线编译结果 |
| `arknights_sim/core/state.py` | profile 后优化 deepcopy 与 canonical snapshot；完整可变状态仍隔离 |
| `arknights_sim/core/event_queue.py` | 明确按时间、事件优先级、语义键、相同事件序号排序 |
| `arknights_sim/core/event_priority.py`（新增） | 集中定义 EVENT_PRIORITY、SIMULATION_PHASES，明确 ASSUMED |
| `arknights_sim/environment/__init__.py`, `env.py`, `action.py`, `legal_actions.py`, `decision_events.py`, `result.py`, `replay.py`, `README.md`（新增） | 独立环境、结构化动作、合法性、决策推进、结果与重放 |
| `agents/__init__.py`, `base.py`, `random_agent.py`, `scripted_agent.py`, `episode.py`, `mcts_agent.py`（新增） | Agent 接口、完整 episode、独立回归基线和搜索入口 |
| `agents/mcts/__init__.py`, `node.py`, `search.py`, `rollout.py`, `heuristic.py`, `stats.py`, `README.md`（新增） | 单人 UCT、两种 rollout、独立评价、统计与计划来源说明 |
| `scripts/run_random_agent.py`, `run_mcts.py`, `replay_solution.py`, `evaluate_agents.py`, `benchmark_mcts.py`（新增） | 可复现实验、保存、重放、多种子对照与性能测量 |
| `tests/test_environment_v02.py`, `test_agents_v02.py`, `test_mcts_v02.py`, `test_replay_v02.py`, `test_event_priority_v02.py`, `test_state_performance_v02.py`（新增） | 环境、搜索、重放、假设规则和复制隔离测试 |
| `golden_tests/README.md` 与四份 JSON；`research/mechanics/` 四份机制说明（新增） | 锁定当前假设行为，保留待校准状态 |
| `pyproject.toml`, `README.md`, `research/BATTLE_ARCHITECTURE.md` | 版本 0.2.0、包含 agents 包、运行方式与事件顺序说明 |
| `outputs/`, `research/v02/`, `STATUS_V02.md`（新增产物） | 方案、原始实验、重放、profile、benchmark、测试报告与本报告 |

没有引入神经网络、PyTorch、AlphaZero/PPO/MuZero、训练缓冲区、UI、客户端操作或多进程。当前依然只使用已支持的黑角、玫兰莎，E0 Lv1、潜能 1、信赖 0、技能等级 1。

## 2. Environment API 与状态

```python
from arknights_sim.environment import ArknightsEnv, Action, WAIT

env = ArknightsEnv(trace=True)
state = env.reset(stage_id="0-1", squad=None, seed=12345)
actions = env.legal_actions(state)
child = env.step(state, actions[0])
terminal = env.is_terminal(child)
result = env.result(child)
copy = env.clone_state(state)
key = env.state_key(state)
next_event_state = env.advance_to_next_decision_event(state)
```

`step` 为纯转移：先确认动作合法，再复制父状态并返回子状态。`squad=None` 加载默认两名干员；显式 squad 为 OperatorData 序列。`state.game` 复用原有完整 GameState，外层 EnvState 添加决策计数、截止条件、上次决策事件和 trace。Core 不依赖 MCTS。

GameState 已保存 clock/time、DP、life、击杀/漏怪/出生计数、地图和 squad、全部单位（含死亡对象）、部署次数、再部署时间、攻击冷却和代数、技能/SP/modifier、wave/spawn 数据、完整 event queue 与 sequence、RNG 和 windup。未部署干员由 squad 减去存活部署实例得到，不建立重复真相源。

clone 对可变字典、单位、技能、modifier 列表、queue、clock、RNG、trace 做深隔离；只有递归证明不可变的数据对象共享身份。`snapshot()` 始终返回脱离原对象的结构。state key 包括 core 的完整哈希、decision_count、max_decisions、horizon；不包含不影响转移的 trace/事件描述。不同调用历史可能有不同键，暂未做最小状态等价化。

相同数据、配置、种子、动作序列在当前平台内 deterministic；未声称跨全部浮点平台或跨模型版本逐位一致。完整接口约定见 [Environment README](arknights_sim/environment/README.md)。

## 3. Action system 与 Legal Actions

`Action` 为 frozen dataclass，可比较、hash、JSON 序列化和可读打印。支持 DEPLOY / ACTIVATE_SKILL / RETREAT / WAIT；方向 RIGHT/UP/LEFT/DOWN，坐标沿用左下角原点 `(col, row)`。文本方案将干员 id 转换为名称。

| 动作 | 必须满足 |
| --- | --- |
| DEPLOY | squad 内、没有存活部署、再部署冷却结束、DP 足够（含重复部署涨费）、部署人数未满、合法可部署类型、格子未被存活单位占用、四向有效 |
| ACTIVATE_SKILL | 存活且已部署、技能存在、manual trigger、SP 足够、当前未激活 |
| RETREAT | 当前存活部署 |
| WAIT | 任一非 terminal 状态均可；terminal 返回空动作集 |

生成器枚举全部兼容格 × 全部四向，没有删除低分动作。搜索只从该集合扩展；episode runner 和 env.step 也校验合法性。每个实际决策的动作数保存在 history 和评估原始行中。

## 4. WAIT 与决策事件

WAIT 的外部语义是“推进至下一个有意义事件”，不是走一帧。内部保留 V0.1 的 60 Hz 物理推进；可选 observer 在 settled tick 边界检测出生/死亡/漏怪、过格、等待点变化、阻挡变化、DP 达到部署门槛、再部署就绪、技能就绪/结束及战斗结束。

普通格内位移、逐帧 HP/冷却变化不会单独打断 WAIT。当前所有敌人过格都算拓扑变化，后续更复杂关卡可能还需减少决策点。非网格队列事件在下一个 settled tick 向 Agent 可见，最多延后一 tick；这是明确的环境约定，并非实测官方操作窗口。

无事件时 WAIT 达到 horizon 后结束；瞬时部署/撤退循环受 max_decisions 限制。默认 300 秒、512 决策。超限为不成功的截断，不能伪装成通关。成功方案只有 43 次决策。ScriptedAgent 使用单独的历史定时桥接 API；MCTS 不使用该桥接。

## 5. Event Priority 与黄金回归

集中优先级：`MODIFIER_BOUNDARY=10`、`SKILL_END=20`、`SPAWN=30`、`HIT=40`。同 timestamp 先按上述类型，再按显式语义 payload 排序；递增序号只用于语义完全相同的事件。不会由不同事件的 Python 插入顺序偶然决定结果。

固定阶段：DP/SP/移动 → 既有队列事件（HIT 内同步处理死亡与释放）→ 阻挡更新 → 干员攻击发起 → 敌人攻击发起 → 新产生零前摇命中 → terminal → 外部动作序列。移动、恢复、死亡和主动技能没有伪造独立 queue 事件，而是按这些明确阶段处理。

该顺序为 ASSUMED。构造的同帧致死交换可能与 V0.1 插入顺序产生不同结果；原有 56 项与原有成功战斗仍通过。四份 golden fixture 是 Simulator 行为回归，不是官方观测金标准。

## 6. Plain Single-Agent MCTS

节点包含 state_key、父节点、入边动作、children、N/W/Q、terminal、untried_actions 和隔离的 state。Tree policy 使用经典 UCT：`Q + c * sqrt(log(N_parent) / (1 + N_child))`。默认 `c=sqrt(2)`，max_depth=64，rollout_limit=160，独立可控 seed；不使用 PUCT。

所有备份从同一 Agent 视角：terminal 零漏怪全部击杀 +1，terminal 不成功 -1；沿 leaf 到 root **不翻转符号**。独立 heuristic 使用击杀进度、life、己方健康，非终局限制在 [-1, 0.95]。一旦漏怪，零漏怪目标不可恢复，rollout 可提前停止并给 -1；这不会把实际 env 强制设成 terminal。

提供均匀 `RandomRolloutPolicy`，以及默认 `TacticalRolloutPolicy`：只按公开 route、阻挡量、朝向后攻击范围、覆盖与支援加权，偏好有意义部署、技能与保留阵地。每个合法动作权重都大于零；扩展采用加权随机排列，不做 pruning。没有 stage==0-1 分支、指定干员/坐标/时间配方。完整未来刷怪数据属于提供的 GameState，本实验是完全可观测的确定性规划。

找到真实成功终局后，Agent 保留搜索产生的最短成功 continuation。每个后续动作先比较预动作 state_key，再确认合法性；偏离则丢弃并重新搜索。该功能可通过 `reuse_successful_plan=False` 关闭。它是**搜索产物复用**，并非每一步重新进行预算次数的搜索。当前保存方案有 1 次新搜索、16 次实际 simulations、17 个树节点，另外 42 次决策复用计划。

Search trace 保存所有根动作（包括 N=0 的未访问动作）、N/W/Q、选择原因和延迟。`found_solution` / `plan_reuse` / `most_visited` 分别标识来源。平均/最大 depth 是树叶深度，rollout 决策数单独统计。实际起点只有 WAIT 合法；大部分完整策略在后续 rollout 中发现，因此不能把浅树误读成全程只考虑两步。

MCTS 的首次成功是在独立于 ScriptedAgent 源码/时间轴的实现阶段得到。运行路径不导入 scripted_agent，测试在新 Python 进程中禁止该 import 仍可搜索。评估器只在独立回归调用中加载 ScriptedAgent，动作历史不传给 MCTS。

## 7. Tests

| 范围 | passed |
| --- | ---: |
| 原有 data / simulator | 15 + 41 = 56 |
| Environment / Action / WAIT | 23 |
| Random / Scripted / import 隔离 | 8 |
| 单人 MCTS / backup / seed / 合法性 | 15 |
| 完整真实方案重放与改动检测 | 13 |
| 同帧优先级 / 四项 golden | 14 |
| clone / snapshot / 性能改动隔离 | 9 |
| 新增合计 | 82 |
| **总计** | **138 passed / 0 failed / 0 skipped** |

新增覆盖 action equality/hash、非法 tile/占位/DP/再部署/技能/撤退、WAIT 事件与上限、clone 与 RNG/trace 隔离、状态键、确定性转移、随机 episode、原方案回归、MCTS 合法扩展/父状态隔离/N/同号备份/+1/-1/相同 seed，以及真实 MCTS 方案全新实例重放。重放文件的动作、时间（含 NaN）、中间键、事件原因、trace、结果、计数遭改动会被拒绝。

完整可机读报告：[test_results.xml](research/v02/test_results.xml)。未用单纯的性能阈值测试，避免共享机器噪声制造假失败。

## 8. Profile 与性能

先运行搜索，再 profile，最后优化。初始 cProfile 在 16 simulations 下主要累积成本为 env.step 11.887s、deepcopy 4.277s、state hash 4.117s、重复 compile_route 2.388s；这些是嵌套成本，不能相加。instrumented 总时间 16.18s 也不能直接与普通运行耗时比较。

只作了明确对应热点的修改：缓存不可变路径；clone memo 共享经递归验证的不可变 setup；canonical 直接遍历字段，省去 asdict 的中间复制；仅为更好/更短的成功候选构造完整 replay keys。没有换语言或引入对象池/多进程。

同状态五批中位数实验：clone 1195 → 2499/s（2.09×）；stable_hash 909 → 1370/s（1.51×），哈希完全不变。详见 [clone_profile.json](research/v02/clone_profile.json)、[初始 profile](research/v02/mcts_initial_profile.txt)。

**完整性能基准尚未运行。** `scripts/benchmark_mcts.py` 已实现并通过编译/CLI 检查，但用户要求当前实验完成后保存并暂停，因此没有继续启动它。events/sec、最终 clones/sec、legal_actions/sec、MCTS simulations/sec/nodes/sec、独立 peak memory 与决策延迟的统一基准暂不填值。

已有 profile 的 clone/hash 同状态比较如上；参数实验中的 wall time、实际 sims/nodes、树深度和逐决策动作数均已保留。它们可描述本轮运行，不替代隔离测量的完整 benchmark。

## 9. 三种 Agent 与 rollout 对照

每一配置固定同样三个 seeds：12345、12346、12347。时间不含初始 GameData 加载；成功率以全部尝试为分母，截断/超时/异常不能排除。每条实验保存 kills、leaks、decisions、wall time、合法动作数、新搜索/实际 sims/nodes、复用次数和最终键。

| Agent / rollout / 每次搜索预算 | 零漏怪通关率 | 平均 kills | 平均 leaks | 平均 decisions | 平均 wall time(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomAgent | 0/3 (0%) | 0.00 | 11.00 | 143.33 | 0.162 |
| ScriptedAgent | 3/3 (100%) | 11.00 | 0.00 | 13.67 | 0.103 |
| MCTS / tactical / 16 | 3/3 (100%) | 11.00 | 0.00 | 43.33 | 2.181 |
| MCTS / tactical / 32 | 3/3 (100%) | 11.00 | 0.00 | 42.00 | 3.965 |
| MCTS / tactical / 64 | 3/3 (100%) | 11.00 | 0.00 | 42.00 | 7.318 |
| MCTS / tactical / 128 | 3/3 (100%) | 11.00 | 0.00 | 42.00 | 14.560 |
| MCTS / tactical / 256 | 3/3 (100%) | 11.00 | 0.00 | 42.00 | 28.981 |
| MCTS / random / 16 | 0/3 (0%) | 0.00 | 11.00 | 142.67 | 11.570 |
| MCTS / tactical / 512 | 3/3 (100%) | 11.00 | 0.00 | 42.00 | 58.172 |
| MCTS / tactical / 1000 | 3/3 (100%) | 11.00 | 0.00 | 42.00 | 115.372 |

ScriptedAgent 只作为历史回归参考，使用历史定时推进钩子，所以其较少的 decisions 不能直接解释为更高规划效率。Random/MCTS 都只使用结构化动作和事件 WAIT。随机 Agent 和均匀 rollout 的失败样本全部保留，没有只展示成功 seed。

样本量每配置只有 3 次；这不是对所有 seed 或关卡泛化的成功率估计。开发机后台负载会带来 wall-time 噪声；独立性能基准以上节为准。最终实验共 30 次，errors=0、timeouts=0。详见 [完整评估 JSON](outputs/agent_evaluation.json)、[逐次 JSONL](research/v02/agent_evaluation.jsonl)、[实验说明](research/v02/AGENT_EVALUATION.md)。

## 10. 搜索强度 vs 通关率与时间

| simulations/search | 成功 / trials | 平均 wall(s) | min–max wall(s) | 平均实际总 sims/episode | 平均 / 最大 tree depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 3/3 | 2.181 | 2.115–2.224 | 16 | 1.938 / 2 |
| 32 | 3/3 | 3.965 | 3.863–4.036 | 32 | 1.969 / 2 |
| 64 | 3/3 | 7.318 | 7.104–7.619 | 64 | 2.203 / 3 |
| 128 | 3/3 | 14.560 | 14.025–15.227 | 128 | 2.602 / 3 |
| 256 | 3/3 | 28.981 | 28.676–29.280 | 256 | 3.391 / 7 |
| 512 | 3/3 | 58.172 | 57.842–58.409 | 512 | 4.548 / 9 |
| 1000 | 3/3 | 115.372 | 111.026–121.264 | 1000 | 5.774 / 10 |

**首次测试到成功的预算是 16**，并非证明最少必需 16；没有为找更低数字额外挑选 seed。512 与 1000 也以相同三个种子运行。全部 tactical 配置均能很早发现成功 continuation，后续沿该搜索产物复用；增加预算主要增加搜索时间，没有在这个小样本上显示更高通关率。

均匀随机 rollout 很容易部署到不覆盖路线的位置或立即撤退，再部署冷却随后限制补救。通用战术 rollout 改善了模拟质量；这没有加入标准答案，但也意味着当前实验**未证明 UCT 相对同一 tactical rollout 单独采样的增益**。未来应增加该消融对照，避免把启发式效果全部归因于树搜索。

## 11. 自主通关方案与完整时间轴

[原始方案 JSON](outputs/mcts_0-1_solution.json) 保存完整设置、逐步动作与 state keys、root 统计及战斗 trace；[可读版本](outputs/mcts_0-1_solution.txt)、[逐决策搜索统计](outputs/mcts_0-1_solution_search.txt)。以下不是输入攻略，是实际搜索产生的输出。

```text
Stage: 0-1
Seed: 12345
Metadata: {"agent": "PlainMCTSAgent", "config": {"mcts_simulations": 16, "exploration_constant": 1.4142135623730951, "max_depth": 64, "rollout_limit": 160, "seed": 12345, "rollout_policy": "tactical", "reuse_successful_plan": true}, "strategy_source": "search-derived only; no scripted input"}

00000.000 WAIT
00003.000 DEPLOY 玫兰莎 @ (7, 2) DOWN
00003.000 WAIT
00005.000 WAIT
00006.017 WAIT
00007.717 WAIT
00016.000 WAIT
00016.950 WAIT
00017.000 DEPLOY 黑角 @ (6, 2) RIGHT
00017.000 WAIT
00018.000 WAIT
00018.650 WAIT
00019.400 WAIT
00019.417 WAIT
00021.650 WAIT
00029.000 WAIT
00029.700 WAIT
00029.867 WAIT
00029.883 WAIT
00030.800 WAIT
00031.567 WAIT
00034.000 WAIT
00034.550 WAIT
00034.567 WAIT
00037.567 WAIT
00045.000 WAIT
00045.700 WAIT
00046.100 WAIT
00046.117 WAIT
00046.783 WAIT
00047.800 WAIT
00050.800 WAIT
00052.000 WAIT
00052.700 WAIT
00053.000 WAIT
00053.583 ACTIVATE_SKILL 玫兰莎
00053.583 WAIT
00053.600 WAIT
00054.283 WAIT
00055.283 WAIT
00058.283 WAIT
00063.700 WAIT
00064.617 WAIT

RESULT: SUCCESS
Kills: 11 / 11
Leaks: 0
Life: 20
Decisions: 43
Wall time: 2.213195s
MCTS simulations: 16
MCTS nodes: 17
Mechanics: four uncalibrated rules remain ASSUMED.
```

核心动作：3.000s 玫兰莎 (7,2) DOWN；17.000s 黑角 (6,2) RIGHT；53.583333333s 玫兰莎主动技能。完整 JSON 保留精确数值，文本显示三位小数。其余为事件 WAIT。坐标按现有底部原点。

## 12. Replay 验证

保存后，新建 Environment/Simulator、重新加载 GameData 与 squad，并从 seed 12345 独立 reset。逐项比对初始键、每个动作时间/合法性/动作数、中间状态键、final result、core hash、environment key、完整 trace hash；存档 trace 本身也核对。

- `identical`: True
- `trace_identical`: True
- 结果：success=True，kills=11，leaks=0，life=20，decisions=43
- 最终 GameState SHA-256：`cf26d870baa2b1a7608b2393223cc54ba5c9e636028fd5b0a33b51b7c19fd469`
- 最终 Environment SHA-256：`424d2704fa326313d0d5973528aec3c9d1184d7fe61866eb6a2a271cce8957aa`

报告：[mcts_0-1_solution_replay.json](outputs/mcts_0-1_solution_replay.json)。这是动作重放格式，默认重建已审阅 loader 参数；自定义属性/synthetic stage 需要 env_factory，初始键不符会拒绝。未提供任意跨版本 snapshot 读档。

## 13. 尚未验证的游戏规则

| 机制 | 当前状态 | 当前简化与尚缺证据 |
| --- | --- | --- |
| [continuous_turning](research/mechanics/continuous_turning.md) | ASSUMED | 逐格中心直线段、瞬时转弯；缺连续轨迹与拐点时刻实测 |
| [blocking_radius](research/mechanics/blocking_radius.md) | ASSUMED | 0.5 格中心距离、tick 检测、固定并列规则；缺体型/偏移/高速穿越测量 |
| [attack_windup](research/mechanics/attack_windup.md) | ASSUMED | 全局默认 0.2s，命中取属性、死亡/代数不符取消；缺角色动画帧与伤害取样验证 |
| [same_frame_priority](research/mechanics/same_frame_priority.md) | ASSUMED | 固定阶段/显式同刻排序；缺官方同帧致死、技能到期与命中并发证据 |

其他未定事项继续保留在 [UNKNOWN_RULES.md](research/UNKNOWN_RULES.md)：随机出生分布、伤害取整/复杂叠加、特殊退场返费、多波/清场门控等。没有因 MCTS 成功而升级为 VERIFIED。`golden_tests/` 只锁定当前模型行为。

## 14. 下一阶段最值得解决的问题

1. 优先采集四项基础机制的独立观测与回归样本，校准路径/阻挡/伤害 timing；避免搜索利用模型误差。
2. 做 tactical rollout-only、关闭成功计划复用的 UCT 等消融，增加 seed 数和可控扰动，测清搜索本身的价值。当前小关卡全部 3/3 无法显示预算曲线差异。
3. 用更繁忙但仍在已支持机制内的场景检查决策事件数量、action branching、horizon/decision 截断，并继续保留所有合法打法。
4. 再依据实际 profile 改善 WAIT 内部观察/状态复制/哈希；目前不需要 C++、Rust、对象池或 multiprocessing。

本阶段不继续添加训练网络或大规模强化学习。下一阶段的训练设计应另行决定，不能把当前未校准的规则默认为正确。

## 15. 复现命令

```bash
.venv/bin/python -m pytest -q --junitxml=research/v02/test_results.xml
.venv/bin/python scripts/run_random_agent.py --stage 0-1
.venv/bin/python scripts/run_mcts.py --stage 0-1 --simulations 16 --seed 12345
.venv/bin/python scripts/replay_solution.py outputs/mcts_0-1_solution.json
.venv/bin/python scripts/evaluate_agents.py --budgets 16 32 64 128 256 512 1000
.venv/bin/python scripts/benchmark_mcts.py
```

全部运行使用本地固定 GameData，不需要 IPA、网络或真实客户端。source/data/输出摘要清单见 [artifact_manifest.json](research/v02/artifact_manifest.json)。

## 16. 暂停点与恢复事项

用户最新指示为“完成当前实验之后保存实验数据并暂停”。已完成正在运行的 1000 simulations 三种子实验，保存完整 16–1000 多种子矩阵、失败对照、138 项测试报告、已有零漏怪方案与 replay，随后停止。未启动 benchmark，也没有开始下一阶段或安排自动继续。

恢复时首先检查本报告与 `research/v02/artifact_manifest.json`；如获新指示，再运行已准备的 `scripts/benchmark_mcts.py`，补齐统一性能表与最终交付核验。现有 MCTS 方案已重放验证，重放无需重新搜索。不要重跑已经完整保存的参数矩阵来替代未运行的性能测试。
