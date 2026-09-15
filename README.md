> 最新：训练样本已改为与模型分开保存，集中放在「训练数据」文件夹。停止训练后可按任务清理样本，模型参数仍保留。见 [保存与清理说明](训练数据/使用说明.txt)。

> 事件规划架构 v3 已整合现有模拟器、双网络、PUCT 和桌面应用；见 [架构与验证报告](STATUS_EVENT_ARCHITECTURE.md)、[审计与修改方案](outputs/event_architecture/AUDIT_AND_PLAN.md)、[逐层参数量](outputs/event_architecture/PARAMETERS.md)。

> 历史：0.5 战斗机制验证版，见 [验证说明](STATUS_MECHANICS_VERIFICATION.md)。

> 奖励机制已更新，保持暂停待验证。最新说明见 [奖励机制验证版](STATUS_REWARD_VERIFICATION.md)。

> 2026-09-15 更新：现已加入职业编队、三种控制、本地全地图和自主编队训练。最新范围与操作见 [STATUS_MAPS_JOINT.md](STATUS_MAPS_JOINT.md)。

# Arknights Zero — AlphaZero Research Environment

**2026-09-15：已导入 PRTS 干员一览全部 431 名干员。** 游戏数据页可搜索并查看资料，模拟器和训练页可使用「全员编队…」选择最多 12 人。全部干员可识别、部署、撤退并进入模型输入；复杂机制仍为显式标注的基础近似模拟。53 名干员的一技能支持基础属性强化。详见 [PRTS 导入与使用说明](STATUS_PRTS_CATALOG.md)。

独立、无 GUI、离线、确定性的 Python 战斗研究引擎。已用真实 **0-1 坍塌（普通模式）** 验证地图、刷怪时序和等待点，并跑通黑角 + 玫兰莎的完整战斗。它是有限机制的研究基线，**尚不是已经校准到实机的世界模型**。

**Simulator V0.2 已提供 Agent Environment、结构化合法动作、按决策事件推进的 WAIT 和单人 UCT MCTS。** 在此基础上新增独立的 Policy/Value Network、神经 PUCT、Self-play、Replay Buffer、AdamW Trainer 和 checkpoint。原有无网络模拟器与 UCT 继续可单独运行。没有客户端自动化或游戏服务器访问。

Simulator 历史结论保留在 [STATUS_V02.md](STATUS_V02.md) 与 [STATUS_REPORT.md](STATUS_REPORT.md)。AlphaZero 的实际实验与限制见 `STATUS_ALPHAZERO_V01.md`。桌面开发先保留为 scaffold；按用户指定顺序，先验证核心训练闭环再接 UI。

搜索使用通用路线/攻击范围启发式 rollout，所有合法动作均保留。找到成功终局后可复用搜索产生的完整计划，逐步验证状态键；复用动作计为零次新模拟。四项底层机制仍是 **ASSUMED**，见 [research/mechanics](research/mechanics)。零漏怪是当前模型中的结果。

上述 rollout/计划复用仅属于原有 **Plain UCT**。新的 **AlphaZero PUCT** 使用神经 priors/value、训练根噪声和访问次数标签，不读取脚本方案或 UCT 的成功计划。网络 Value 为 [-1,1] 的预测值，不是经过校准的通关概率。

## AlphaZero

运行时依赖为可选组，保持核心 Simulator 不依赖 PyTorch 或 Qt：

```bash
.venv/bin/python -m pip install -e '.[test,alphazero]'
.venv/bin/python scripts/benchmark_alphazero.py
.venv/bin/python scripts/train_alphazero.py --iterations 3 --episodes 5 --simulations 16 --steps 5 --eval-episodes 3 --device cpu --run-dir outputs/alphazero/smoke
.venv/bin/python scripts/plot_training.py outputs/alphazero/smoke/training_metrics.jsonl
.venv/bin/python scripts/inspect_policy.py --checkpoint outputs/alphazero/smoke/checkpoints/best.pt --simulations 32
```

默认完整配置在 [configs/alphazero_001.yaml](configs/alphazero_001.yaml)。正式运行 `scripts/train_alphazero.py` 会自动产生对局、训练、固定种子评估和 checkpoint。使用新的 `--run-dir` 隔离实验；`--resume latest.pt --iterations N` 继续训练至总计 N 轮。Ctrl-C 请求安全停止，不给中断的对局填造结果。

`latest.pt` / `iteration_*.pt` 包含模型、optimizer、训练样本引用、RNG、未完成 iteration 进度；训练样本按任务分批存放在 `训练数据/`，不会整套重复写入每份模型存档。`best.pt` 是最佳评估模型的独立推理快照。恢复训练用 latest 或 iteration 文件；样本缺失时会提示并重新积累样本，保留模型参数和 optimizer。旧格式仍按原有版本兼容规则读取，再次保存时采用分离格式。详细接口见 [network/README.md](network/README.md)、[training/README.md](training/README.md)。

## 运行

Python 3.11+。当前工作区已建好 `.venv`（Python 3.12）；模拟器运行本身只依赖标准库。

```bash
.venv/bin/python scripts/inspect_stage.py level_main_00-01
.venv/bin/python scripts/validate_stage.py
.venv/bin/python scripts/run_battle.py --no-operators --trace research/movement_trace.txt
.venv/bin/python scripts/run_battle.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_random_agent.py --stage 0-1
.venv/bin/python scripts/run_mcts.py --stage 0-1 --simulations 16 --seed 12345
.venv/bin/python scripts/replay_solution.py outputs/mcts_0-1_solution.json
.venv/bin/python scripts/evaluate_agents.py --budgets 16 32 64 128 256 512 1000
.venv/bin/python scripts/benchmark_mcts.py
```

新环境可执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

数据已经下载并固定在 `data/real/`，以上命令无需联网、无需 IPA。`data/manifest.json` 包含源提交和 SHA-256。研究更新脚本 `fetch_*.py` 才会读取公开 GitHub；不会访问游戏服务器。`recon.py` 只读用户原 IPA，不影响模拟器运行。

## Agent API

```python
from arknights_sim.environment import ArknightsEnv, Action, WAIT
from agents.mcts import MCTSConfig
from agents.mcts_agent import PlainMCTSAgent

env = ArknightsEnv(trace=True)
state = env.reset(stage_id="0-1", seed=12345)  # 默认黑角、玫兰莎 E0 Lv1
agent = PlainMCTSAgent(MCTSConfig(mcts_simulations=16, seed=12345))
while not env.is_terminal(state):
    action = agent.select_action(env, state)
    assert action in env.legal_actions(state)
    state = env.step(state, action)  # 返回隔离的子状态，父状态不变
print(env.result(state))

# 动作示例，执行前必须确认在 env.legal_actions(state) 中：
action = Action("DEPLOY", "char_208_melan", (3, 4), "LEFT")
# WAIT 自动推进至出生、过格、阻挡变化、费用满足、技能就绪等决策事件。
# env.clone_state(state)、env.state_key(state)、
# env.advance_to_next_decision_event(state) 均可独立使用。
```

`reset(squad=...)` 接受由 `OperatorLoader` 加载的 `OperatorData` 序列。`state.game` 保留完整原有 `GameState`；Environment 的决策数、截止时间及 trace 位于外层。详见 [environment/README.md](arknights_sim/environment/README.md) 与 [MCTS 说明](agents/mcts/README.md)。

输出完整动作及战斗 trace：[JSON 方案](outputs/mcts_0-1_solution.json)、[可读时间轴](outputs/mcts_0-1_solution.txt)、[逐决策搜索统计](outputs/mcts_0-1_solution_search.txt)。

## 原有 Simulator API

```python
from pathlib import Path
from arknights_sim import Simulator
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader

root = Path.cwd() / 'data' / 'real'
stage = StageLoader(root / 'enemy_database.json').load(root / 'level_main_00-01.json')
loader = OperatorLoader(root / 'character_table.json', root / 'range_table.json',
                        SkillLoader(root / 'skill_table.json'))
squad = [loader.load('char_500_noirc'), loader.load('char_208_melan')]
sim = Simulator(stage, squad, seed=12345, trace=True)
sim.run_until(4)
sim.deploy('char_500_noirc', (3, 2), direction=0)
sim.run_until(17)
sim.deploy('char_208_melan', (1, 2), direction=0)
sim.run_until(67)
sim.activate_skill('char_208_melan')
child = sim.clone()
child.step(1)  # 不修改父节点
sim.run(180)  # 到绝对时间上限；战斗结束会提前停止
print(sim.state.result, sim.state.killed, sim.stable_hash())
```

- 坐标 `(col, row)` 从左下角起算；方向 `0/1/2/3` = 右/上/左/下。
- `run_until(t)` 使用绝对时间，`step(seconds)` 使用相对时长。动作边界必须对齐 dt 网格，默认 1/60 秒；不对齐报错。内部 spawn/hit/skill 事件可用非网格时间。
- `GameState.clone()/snapshot()/stable_hash()/step()` 也可独立使用；snapshot 是 JSON 可序列化调试结构，暂未提供跨版本读档。
- `retreat(id)` 与 `add_modifier(Modifier(...))` 是额外动作。返费、碰撞、索敌等简化政策都有独立测试和未知规则编号。
- trace 可关闭；开启后 `sim.trace.text()` 输出按时间稳定排列的日志。

## 模块

`data/` 将原始表转成不可变领域模型；`map/` 做路径规划；`entities/` 保存可变单位状态；`combat/` 独立处理伤害、索敌、阻挡和攻击事件；`skills/` 管 SP 与 modifier；`deployment/` 处理费用；`core/` 包含状态、时钟与事件队列；`debug/` 输出 trace；`environment/` 暴露纯状态转移与合法动作；顶层 `agents/` 实现随机、回归脚本和 MCTS。Core 不依赖 Agent。

## 已知限制

历史严格回归仅覆盖黑角/玫兰莎 E0 Lv1、潜能1、信赖0、技能等级1。新增全员模式默认同等级，支持基础普攻/治疗与部分技能，未实现的技能禁用，天赋/模组/复杂特性不作完整还原。只接受经过审阅的简单近战步行敌人和单波无条件刷怪。复杂技能、天赋、远程敌人、飞行、地形机关、清场门控、多波、剧情交互、突袭符文会被拒绝或明确排除。1-1 原始文件用于结构研究，不列为完整战斗验收关卡。

0-1 的出生/目的地、等待与刷怪时间已核验；连续转弯轨迹、随机出生分布、0.5 格阻挡半径、0.2 秒攻击前摇、同帧优先级是待校准策略。没有实机逐帧回归数据，不能把 11 杀结果解释为真实客户端必然同样通关。无干员漏 11 个但仍有 9 耐久，Core 结果是 WIN；Environment 要求全部击杀且零漏怪才记 success，不实现游戏星级。

第三方源码和 GameData 的权利边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
