# Arknights-unity 整合评估

检查版本：`3c81f83d58711b27e8bd29ac0d0e5c468362b591`（master），2026-09-15。
结论：可作实现参考和交叉检查对象，不建议整体整合或替换 Python 模拟器核心。当前没有移植其运行时代码。

## 实际检查范围

GitHub 完整树（truncated=false）、README、仓库元数据、Assets/Script 下 26 个 C# 文件、Unity 版本和包清单。源码仅保存在 research/integration/arknights_unity/source 作审阅快照，未加入执行路径。未下载官方美术/动画资源，未启动 Unity，未进行运行时性能或正确性验证。

## 关键证据

- Unity 2022.3.56f1c1，C#；核心 Char/Enemy/BattleController 都是 MonoBehaviour，依赖 GameObject、Transform、Time.deltaTime、物理 Trigger、Spine 动画和 DOTween。当前项目为 Python GameState、显式时钟与事件队列，不能直接 import 这些脚本。
- CharData.ParseCharData 只接受有 Resources/Prefab/Battle/Char/Char<ID> 预制体的干员；该目录实际只有 Char103、Char263、Char367、Char502 四个预制体。加载数值固定取最后精英阶段最后属性帧，不能替代当前逐等级/潜能配置。
- 已检查项目脚本没有找到完整技能解析、SP/技能激活系统或召唤物生命周期实现。不能用它补齐全干员技能或令的合体、缪尔赛思复制等能力。
- Enemy.EnemyMoveController 到终点只注释“扣血”并 Destroy；Enemy.OnDestroy 无条件 killNum++；BattleController 以 killNum==enemyNum 判定结束。这个路径混淆到达终点与被击杀，不能借来验证 0 漏怪。
- Char.TakeDamage / Enemy.TakeDamage 的死亡条件是 hp<0，精确归零边界不完善。Char 的伤害 switch 没有 Real 分支。
- EnemyController 将 WAIT_FOR_SECONDS / WAIT_FOR_PLAY_TIME / WAIT_CURRENT_FRAGMENT_TIME / WAIT_CURRENT_WAVE_TIME 合成同一个相对 waitTime，不能作为这些不同时间语义的正确实现。
- EnemyMove 使用 A* Pathfinding Seeker 回调、路点阈值和逐渲染帧位移；未见 GameState clone、稳定 state key 或确定性重放接口。移植需显式状态化与时钟改造，不能靠 Unity 无头运行自动获得这些能力。
- BuffController 仅有敌方 Originium 行为，友方分支为空。敌方每帧 180*deltaTime 真伤、攻击倍率 +0.5，Trap/Originium 设置 300 秒。这是可调查线索，不是官方周期/数值/叠加规则的验证证据。
- Box 放置后重新扫描路径，可以参考障碍物导致重寻路的交互场景；现有 Python 核心已有障碍物重寻路，不需要用 Unity 实现替换。

## 授权

目标 fork 与上游 GitHub metadata 的 license 均为 null，完整目录未发现覆盖项目自有代码的顶层 LICENSE；存在的第三方 license 不能覆盖自有代码。README 写“所有内容均来自 prts.wiki 和游戏解包，仅供学习使用”，这不是清晰的软件复制、修改、分发授权。
因此暂不把源码复制/翻译进 simulator runtime 或打包应用。若以后确实要移植具体实现，需要确认对应作者及第三方授权范围。可以独立研究公开规则并编写自己的实现。未联系作者。

## 建议整合路线

1. 保留现有 Python 核心、真实 GameData Loader、确定性状态与回放接口。
2. 将仓库作为参考资料：障碍物重寻路、出现/消失路点、持续地形效果等形成待验证机制清单。
3. 每个机制先用真实数据和独立资料确认语义；不要从此复刻项目直接认定为 VERIFIED。
4. 在现有 mechanics/combat/skills/summons 模块独立实现；覆盖状态复制、事件优先级、时间边界和重放测试。
5. 若授权明确且某段代码确有优势，再逐模块移植算法，保留出处和授权声明，避免引入 Unity UI/动画依赖。
6. 另一种选择是独立 Unity 后端通过进程协议连接桌面，但当前仓库缺少无图形 API、克隆、技能覆盖与精确结果，不建议为其建立此后端；M1 8GB 的实际资源成本也未测量。

## 当前项目状态

本次仅完成用户新提出的整合评估和参考资料保存。之前暂停的召唤物实验没有恢复；仍有 23 passed / 1 failed 的未验收版本，详见 STATUS_SUMMONS_PAUSED.md。本次不运行模拟器测试，因为未修改产品运行时代码。

本项目不构成“100%所有干员/地图/作战机制”的现成解决方案。
