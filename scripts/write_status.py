from pathlib import Path
import json, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
b = json.loads((R / "benchmark.json").read_text())
movement = json.loads((R / "movement_result.json").read_text())
battle = json.loads((R / "battle_result.json").read_text())
ipa = json.loads((R / "ipa/facts.json").read_text())
tests = ET.parse(R / "test_results.xml").getroot()
suites = list(tests.iter("testsuite"))
n = sum(int(s.attrib.get("tests", 0)) for s in suites)
failed = sum(
    int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites
)
(ROOT / "STATUS_REPORT.md").write_text(f"""# Arknights Zero — Phase 1 状态报告

日期：2026-09-15。交付：**可运行的 Simulator V0.1 与研究证据集**。真实 0-1 已完成数据/地图/刷怪/等待核验和完整战斗演示；战斗精度仍是明确标注简化的研究基线，不是已逐帧校准的 AlphaZero 世界模型。

## IPA

- 文件：`{Path(ipa["file"]).name}`，{ipa["size"]:,} 字节。
- Bundle `{ipa["bundle"]}`，版本 {ipa["version"]}，Build {ipa["build"]}。
- CONFIRMED：Unity 2021.3.39f1、IL2CPP metadata version 29、ARM64。
- 找到 UnityFramework、Managed/Metadata/global-metadata.dat、Unity player 数据、3,085 个 .ab、13 个 JSON。
- metadata 中直接观察到 BattleController、BattleFactory、Enemy、TargetSelector、AttackState、Projectile、Waypoint 等语义名称；没有据名称推导具体函数行为。
- 主程序和 UnityFramework 的 cryptid=0 仅描述当前提供文件，未推断其来源或历史。没有修改 IPA、执行客户端、绕过加密/签名/反篡改、连接游戏服务器或提交 Battle Record。
- 完整路径清单、二进制事实、跳过项目分别见 [IPA_STRUCTURE](research/ipa/IPA_STRUCTURE.md)、[ENGINE_ANALYSIS](research/ipa/ENGINE_ANALYSIS.md)、[SKIPPED_FILES](research/ipa/SKIPPED_FILES.tsv)。原文件 SHA-256 已记录。

## GitHub

实际读取 5 个指定仓库的元信息、目录树与相关 README/源码：djpadbit/Arknights-RE、DD-Channel/Arknights-Re-Engraved、Kengxxiao/ArknightsGameData、winny727/ArknightsMapViewer、christwsy-zz/Arknights-Simulation。

补充检查 Saukiya/Arknights 和 insomnyawolf/Arknights 的元信息与关联说明；搜索发现 Wirowo/ArknightsBattleSimulator，因其依赖 MITM/真实客户端排除。没有执行任何第三方研究脚本。

关键发现：MapViewer 的 TimelineSimulator 是随机占位，不能用作时序标准；Arknights-Simulation 的 AbstractGame 为空。社区复刻可以帮助理解模块分层，但不应假设它已准确还原规则。MapViewer GPL-3.0，其他主仓库根许可证未检测到，均未复制实现进入引擎。详见 [GITHUB_RESEARCH](research/GITHUB_RESEARCH.md) 与 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

## Battle architecture

已建立 Stage/Map/Tile、Unit/Operator/Enemy、Movement、Targeting、Blocking、Combat、Skill/SP、Modifier、Deployment/DP/Redeploy、EventQueue/Clock、GameState/Trace 模块。

原始 JSON 只进入 Loader；Simulator 消费不可变领域模型。状态可 deepcopy 克隆，snapshot 可 JSON 序列化，stable_hash 包括数据、队列、时钟和 RNG。未来事件驱动扩展已留边界；当前采用 60 Hz 检测与精确时间事件混合。

架构各项证据等级见 [BATTLE_ARCHITECTURE](research/BATTLE_ARCHITECTURE.md)。没有开发 AlphaZero、MCTS、训练器或 GUI。

## GameData

固定 GameData 提交：`0ef7f952dfd018392200157a5c79a6511ba69122`。7 个原始 JSON 文件 SHA-256 全部核对一致，未修改原始数据。

| 文件 | 本阶段用途 |
|---|---|
| level_main_00-01.json | 主验收真实关卡 |
| level_main_01-01.json | 双路线/等待点等补充结构研究，未列入完整战斗支持 |
| enemy_database.json | enemy level 继承和 defined 覆盖 |
| character_table.json | 黑角/玫兰莎 E0 Lv1、潜能1、信赖0 |
| skill_table.json | 玫兰莎 S1 Lv1：50 SP、20秒、ATK +10% |
| range_table.json | 真实攻击范围格与方向旋转 |
| battle_misc_table.json | 已阅读关卡与场景映射结构，不作为完整战斗公式来源 |

来源链接、提交和摘要见 [data/manifest.json](data/manifest.json)。IPA 和该公开快照不是同版本一致性证明。

## Stage 与交叉验证

真实关卡：**0-1 坍塌，普通模式，9×6，11 敌人，10 个 route entry 中 7 条用于刷怪**。

PRTS.Map V0.49.60 正常站内访问成功。地图红蓝门、高台布局、7 组刷怪起始时间/数量/间隔、士兵属性、地图费用参数、3秒等待与本地解析相符。8 项机器对照全部通过。观察记录不是从 parser 反向生成；见 [STAGE_VALIDATION](research/STAGE_VALIDATION.md)、[独立读数](research/prts_map_observation.json)、[对照结果](research/stage_validation.json)。

连续转弯几何仍与查看器平滑线存在差别，未量化实机移动秒数；该部分仅达成拓扑一致，不能宣称逐帧一致。

## Simulator 支持与演示

已运行：刷怪、路点寻路、等待、到达蓝门、部署、撤退、再部署、DP、阻挡配额及释放、双方索敌与攻击、物理/法术/真实伤害、死亡、基础治疗函数、主动技能、自回SP、限时增益、可选 trace、确定性与状态克隆。

- 无干员：{movement["spawned"]} spawn / {movement["escaped"]} escape，剩余 {movement["life"]} 耐久，模型时间 {movement["time"]:.3f} 秒。
- 黑角 + 玫兰莎：{battle["killed"]} kill / {battle["escaped"]} escape，剩余 {battle["life"]} 耐久，模型时间 {battle["time"]:.3f} 秒，结果 {battle["result"]}。
- 动作：4秒黑角(3,2)朝右；17秒玫兰莎(1,2)朝右；67秒开启S1。
- 无干员剩余耐久大于0，因此基础判定也是 WIN；未实现星级。战斗输出不是实机重放。

日志：[movement_trace.txt](research/movement_trace.txt)、[battle_trace.txt](research/battle_trace.txt)。

## Tests

**{n - failed} passed / {failed} failed**。包含真实数据解析、地图方向、路点/等待、刷怪精确时刻、伤害边界、阻挡配额、索敌并列、双方攻击、死亡释放、费用/再部署、技能起止、modifier、克隆/哈希/RNG、同动作确定性和完整真实关卡演示。

额外验证待命中事件取消、再部署代数隔离、step 分段不影响结果、trace 开关不影响状态，以及无支持机制明确报错。JUnit 结果见 [test_results.xml](research/test_results.xml)。测试证明当前模型的行为，不替代未完成的实机一致性验证。

## Benchmark

Python {b["python"]}，{b["platform"]}；无干员0-1、trace关闭、10次运行。

| 指标 | 本次 baseline |
|---|---:|
| 每次模拟时间 | {b["simulated_seconds_per_run"]:.3f} 秒（到终局） |
| 每次平均墙钟时间 | {b["mean_wall_seconds"]:.6f} 秒 |
| simulated seconds / real second | {b["simulation_seconds_per_real_second"]:.1f}× |
| events / second | {b["events_per_second"]:,.0f} |
| state clone / second | {b["state_clones_per_second"]:,.0f} |

事件数包括固定 tick 与显式队列事件；运行计时不含读取 GameData 和实例初始化。clone 使用20秒处状态、200次 deepcopy。该基线不是战斗密集场景或 MCTS 吞吐承诺。原始测量见 [benchmark.json](research/benchmark.json)。

## Unknown Rules 与验收边界

10项规则已逐项登记 Question / Known evidence / Hypothesis / Confidence / Needed verification，见 [UNKNOWN_RULES](research/UNKNOWN_RULES.md)。关键未定项包括连续寻路、随机偏移、阻挡几何、同帧优先级、攻击前摇/弹道、复杂伤害取整/叠加、再部署返费边界和多波清场门控。

V0.1 的功能性验收已跑通；“路线基本一致”限于地图、逻辑路点和等待，连续轨迹仅部分满足。尚无实机逐帧对照集，因此不将本阶段标记为高保真世界模型验收通过。复杂敌人/技能、多波、机关与条件调度拒绝或不纳入范围。

## 下一阶段最值得研究的5个问题

1. 以0-1公开录像校准移动倍率、直线/转弯轨迹、出生随机偏移和到达判定。
2. 测定黑角/玫兰莎攻击前摇、技能开关是否重置攻击、取消与命中时属性快照。
3. 用相邻干员与同时入范围敌人验证阻挡半径、索敌优先级和同帧事件次序。
4. 选择包含清场门控的第二关，建立动态 wave/fragment scheduler，确认 postDelay 与 maxTimeWaitingForNextWave。
5. 获取独立伤害/SP/DP边界实验，建立 golden trace 后再考虑固定点数值和完全事件驱动优化。

## 快速复现

```bash
.venv/bin/python scripts/inspect_stage.py level_main_00-01
.venv/bin/python scripts/validate_stage.py
.venv/bin/python scripts/run_battle.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/benchmark.py
```

当前工作区已有 Python 3.12 虚拟环境；运行命令不需要联网、IPA 或真实客户端。详细 API 和支持范围见 [README](README.md)。
""")
