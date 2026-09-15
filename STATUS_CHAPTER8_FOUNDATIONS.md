# 第八章与账号近卫：当前实现进度

**尚未完成全部目标。** 17 个普通战斗关卡已逐项审计；R8-1 基础战斗已可运行。其余 16 关、71 名账号近卫的完整技能/天赋/模组仍有缺失，正式训练保持关闭，没有用默认 0-1 或简化技能替代账号训练。

## 账号

用户已明确确认视频转录，并补充乌尔比安为游戏一潜（内部 potential=0）。固定快照：
`data/accounts/snapshots/82d3e5dbea0c91b94275c195f5aac466f19115a81e86af65d1b0204ad5e49408.json`。

71 名近卫，信赖按用户要求统一 200%。新增账号战斗输入适配器，读取各自等级、精英、潜能、技能等级和专精；拒绝未持有、未解锁技能或篡改后的快照。已能绑定 101 个已解锁技能选择的基础输入，**绑定不等于技能效果实现**。装备模组缺失效果仍明确显示。

修复信赖属性遗漏：按 favor_table 的 percent → battlePhase → favorKeyFrames 换算，不能把 200% 直接当作属性插值等级。乌尔比安满信赖实际增加 HP 500、ATK 80；100% 与 200% 基础信赖属性相同。

## 刷怪研究

本阶段普通战斗目录共 17 关：R8-1 至 R8-11、M8-6 至 M8-8、JT8-1 至 JT8-3。跳过剧情、突袭和 H8。

- R8-8 存在 2 个 blockFragment 组；清场前不能推进依赖片段。
- M8-8 为 3 波、JT8-3 为 4 波；各波清场与延时分别处理。
- 不能将全部第八章出怪统一改为清场触发。真实数据仍含 wave/fragment/action 延时和组内间隔。
- 现有事件调度已保留这些关系，死亡后放行清场门槛；原 0-1 时间轴回归保持通过。
- 新增相对波次/片段等待点，等待截止取已激活时钟；过期不重新等待。CURRENT 跨片段绑定语义仍为 ASSUMED，未宣称实机验证。

逐组原始数据、来源校验值及独立组件失败原因保存在 `outputs/guards_chapter8/scenarios/capability_audit.json`；摘要见同目录 `capability_audit.md`。复查入口：`python scripts/audit_chapter8.py`。

## R8-1 与道路障碍物

新增道路障碍物：HP 8000、DEF 200、RES 20、阻挡数 0；我方可攻击、低于普通敌人的选取优先级、不可手动撤回。击毁不计敌人击杀、不扣生命，解除格子占用并重新规划余下路线。

修复装置实例错误使用默认 100 HP，而非配置 HP 的问题。新增护送目标 0.6 范围束缚检测，但护送单位完整规则尚未开放，不能视作护送机制验收。

R8-1 共 56 个声明敌人、5 个道路障碍物。已更新地图目录，使现有 Simulator 入口可选此关。空队环境测试：111 次 event-driven 决策，5 漏怪后正常失败；全新实例重放，状态键、动作历史、完整 trace 一致。这是运行与确定性证据，**不是通关或学习证据**。

证据：`outputs/guards_chapter8/scenarios/r81_environment_smoke.json`。

## 测试

修改前 420 passed / 0 failed，68.04 秒。
本轮新增 13 项基础回归，完整测试 433 passed / 0 failed，66.45 秒。
结果：`outputs/guards_chapter8/tests_foundations.xml`。

覆盖信赖换算/上下界、账号绑定/篡改、相对等待、克隆隔离/重放、障碍物伤害/选取/死亡计数、真实 R8-1 运行。

## 主要新增与修改

- `arknights_sim/data/trust.py`：真实信赖属性换算。
- `accounts/battle_inputs.py`：固定账号练度绑定。
- `arknights_sim/data/operator_loader.py`、`data/models.py`：信赖输入和状态。
- `arknights_sim/core/stage_events.py`、`core/simulator.py`、`map/route.py`、`data/stage_loader.py`：相对等待及战斗接入。
- `arknights_sim/mechanics/definitions.py`、`mechanics/runtime.py`：道路障碍物。
- `arknights_sim/data/map_catalog.py`：修复无编号剧情记录导致按关卡编号查找崩溃。
- `data/maps/catalog.json`：仅开放 R8-1 基础模拟；不等于正式训练认证。
- `scripts/audit_chapter8.py`、`scripts/audit_account_guards.py`、`tests/test_chapter8_foundations.py`。
- `research/mechanics/chapter8_roadblocks_and_waits.md`：证据与假设。

## 尚未解决，不能静默跳过

- 特殊友军计数、平民保护目标、斗士塔露拉、预置迷迭香/盾卫、矿石/冰晶友方装置。
- DISAPPEAR/APPEAR_AT_POS、飞行、branches、PLAY_OPERA 及其实际战斗副作用。
- 帝国炮火先兆者、突击者/盾卫、梅菲斯特、塔露拉/黑蛇等特殊攻击、阶段转换及关卡目标。
- 账号已解锁技能中，90 个独立技能 ID 仍被技能 Loader 拒绝；被接受的其余技能也未等同于完整干员校准。特性、天赋、模组仍缺覆盖。
- 玩家观测未来事件隔离、账号选队/选技能的训练阶段绑定、模型存档与快照固定关联尚未完成。
- 连续转弯、阻挡半径、攻击前摇与同帧优先级仍保持既有未完全验证状态。

本次未启动长期训练，也没有生成“第八章已学会”的模型。下一步必须继续完善保护目标/特殊友军与路线，以及账号能力处理器，再逐关解锁；不能以本报告替代全部关卡和全近卫验收。
