from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'research/sources';R=ROOT/'research'
projects=[
('djpadbit/Arknights-RE','公开客户端与格式研究','README.md; DNFBDmp/（仅目录定位，未运行）','Unity、IL2CPP、Lua 桥接与热更新的公开描述。没有提供已验证的完整战斗模型。','AssetBundle 容器与内部 TextAsset 需区分；旧 JSON/BSON 与 FlatBuffers 并存的历史描述。','只采纳格式/架构描述。不执行解密、MITM、登录或 battle record 工具。本文读到的是跨版本研究，不能代替本 IPA 的事实。'),
('DD-Channel/Arknights-Re-Engraved','Unity 社区复刻','Assets/Arknights/Game/Entity/{Entity,Char,Monster}.cs; Game/TargetSelector.cs; Game/Dungeon.cs','Entity 包含集中伤害入口；Monster 有移动/攻击状态、阻挡配额及释放；TargetSelector 使用方向化范围和可替换排序；Spine OnAttack 事件触发命中。','Data/Char、Data/Monster、Data/Dungeon 是社区适配层，并非直接兼容本次全部 GameData。','README 指向 Saukiya 与 insomnyawolf；游戏素材明确归原权利人。此处 C# 更像社区重写，但不能证明每个文件来源；第三方插件与资源不可归入自有代码。'),
('Kengxxiao/ArknightsGameData','公开游戏数据快照','zh_CN/gamedata/levels/obt/main/level_main_00-01.json; levels/enemydata/enemy_database.json; excel/{character,skill,range}_table.json; battle/battle_misc_table.json','支持模型字段与数值；无法单靠数据证明碰撞、伤害结算顺序或攻击前摇。','真实地图索引、tiles、routes/checkpoints、waves/fragments/actions、enemyDbRefs、m_defined 属性覆盖。','这是官方数据的社区镜像，不是可随意重新授权的算法库。锁定提交与文件摘要，运行只读取本地数据。与 IPA 的数据版本一致性 UNKNOWN。'),
('winny727/ArknightsMapViewer','Windows 地图查看器','ArknightsMapViewer/ArknightsMap/{LevelData,MoveRoute,StageData}.cs; View/Drawer/{MapDrawer,RouteDrawer}.cs; View/TimelineSimulator.cs; Utils/PathFinding.cs','只作为地图/路线工具。TimelineSimulator 构造函数使用随机 100–200 秒和随机波数的占位逻辑，不是刷怪时序 oracle。','LevelData 对原始数组做 height-row-1 翻转；MoveRoute 将等待附着在前一位置；路点间寻路及平滑；MapDrawer 再翻转成屏幕坐标。','GPL-3.0；仅本地阅读其结构与语义，未把 C# 代码翻译/复制进 Python 包。未运行 Windows GUI，不宣称逐像素/逐路径完全一致。'),
('christwsy-zz/Arknights-Simulation','早期 agent/战斗模拟框架设想','ArknightsSimulationCore/Bases/{AbstractGame,AbstractEnemy}.cs; Bases/Agent/AbstractDpsAgent.cs; README.md','AbstractGame 为空；AbstractDpsAgent 暴露 Attack(List<AbstractEnemy>)。不能当作完成的世界模型。','README 提出 level editor/team creator；没有验证当前真实关卡格式的证据。','2019 年早期项目，概念参考价值高于实际规则证据。许可证缺失，不能默认复制。')]
text='# GitHub 公开研究\n\n检查日期 2026-09-15。公开 API/原始文件均只读。提交号以实际 tree 响应为准；没有运行任何第三方仓库脚本。\n\n'
for repo,purpose,files,battle,data,risk in projects:
 name=repo.split('/')[-1];m=json.loads((S/(name+'.json')).read_text());t=json.loads((S/(name+'_tree.json')).read_text());lic=(m.get('license') or {}).get('spdx_id','UNKNOWN / 未检测到根许可证')
 text+=f'''## [{repo}](https://github.com/{repo})

- Purpose: {purpose}
- Language: {m.get('language') or 'UNKNOWN'}（GitHub repository API）
- License: {lic}。目录树中的第三方许可证不自动覆盖主项目。
- Last known relevance: pushed_at `{m.get('pushed_at')}`；读取 tree SHA `{t.get('sha')}`。更新时间不等同战斗规则准确度。
- Useful files: {files}
- Useful battle-system concepts: {battle}
- Useful game-data concepts: {data}
- Can reuse directly?: **否，未将任何仓库实现作为运行依赖或拷贝进模拟器**。公开数据另按原权利归属管理。
- Should reimplement?: 是；独立 dataclasses、队列、寻路、状态转移与测试。数学规则和结构描述与实现代码分开。
- Risks / uncertainties: {risk}

'''
text+='''## 相关复刻与搜索结果

读取了 Saukiya/Arknights 与 insomnyawolf/Arknights 的公开仓库元信息，以及 Re-Engraved README 对两者的说明。二者均为 C# Unity 复刻，根许可证未被 API 检测到；未进一步审计其全部代码。前者最后 push 2022-01-21，后者 2023-10-10。不能据 README 确认所有源码都由社区原创。

GitHub Repository Search 实际执行四条查询：`Arknights battle simulator`、`Arknights simulator`、`Arknights combat simulation`、`Arknights level simulator`，原始响应保存在 sources/search_*.json。后两条本次未得到结果；泛词检索中有大量抽卡、基建和账号脚本，与目标无关。

额外检查 [Wirowo/ArknightsBattleSimulator](https://github.com/Wirowo/ArknightsBattleSimulator) 的 README 与元信息。它依赖 mitmproxy、模拟器和真实客户端，名称虽是 Simulator，实际不是独立 battle engine；直接排除，不运行。仓库 API 未识别语言/许可证，最后 push 2022-10-17。

Web Search 通过 Google 公开搜索页检索，页面 HTTP 200 已保存，但未将搜索页面本身视为规则证据。PRTS wiki 的伤害和攻击间隔页面本次 HTTP 403；没有绕过访问限制。随后通过浏览器正常站内导航，成功读取 PRTS.Map V0.49.60 的 0-1 页面；直接 HTTP 请求该 SPA 深链接返回 404，但站内点击可以访问。

## 跨来源结论

1. IPA 与 Arknights-RE 支持 Unity/IL2CPP 架构判断；Lua/FlatBuffers 具体资源版本仍未知。
2. GameData 与 MapViewer 对地图坐标、路点、等待字段的解释相符。
3. PRTS.Map 的 0-1 地图、7 组出兵时间、数量、间隔及士兵属性与本地数据一致，详见 STAGE_VALIDATION.md。
4. Re-Engraved 的物理/法术最低 5% 伤害与公开常见公式一致，但此次 PRTS wiki 访问受限，尚缺独立实机数值验证。该公式在本项目标 HIGH CONFIDENCE，复杂叠加 UNKNOWN。
5. Re-Engraved 用动画事件触发伤害，提供“攻击开始不应自动等于命中”的结构证据；不能据此确定本 IPA 中具体前摇秒数。

## 源码与数据边界

sources/ 下保存的是本地阅读证据，原版权与许可证持续适用；这些文件不在 Python package 发现范围内。新引擎没有导入或调用第三方项目实现，也没有移植其控制流。这里采用的是“读公开规则后独立实现”，不是宣称已经完成法律意义上的双团队隔离 clean-room 审计。不对游戏数据或第三方源码赋予新许可证。
'''
(R/'GITHUB_RESEARCH.md').write_text(text)
