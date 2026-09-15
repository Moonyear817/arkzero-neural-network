# Arknights Zero Desktop 0.6 — 简洁工作区

2026-09-15。已按确认的方案修改现有应用。验证实验已停止，数据和检查点已保存。

## 现在怎样使用

主导航为 **训练、对战、模型**；底部为 **研究工具、设置**。

| 页面 | 默认显示 | 按需展开 |
| --- | --- | --- |
| 训练 | 关卡、编队、训练起点、实验规模；一个主按钮；迭代、评估通关率、耗时和一张曲线 | 高级配置与 YAML、更多操作、详细指标与图表 |
| 对战 | 关卡、控制模式、编队、播放速度、大地图；开始／暂停／继续主按钮 | 战斗时间轴、种子和搜索等更多设置；暂停时显示单步 |
| 模型 | 简明列表、用此模型对战、评估 | 加载研究模型、比较、完整元数据与结果 |
| 研究工具 | 搜索分析、游戏数据、日志三个标签 | 保留已有工具 |
| 设置 | 界面语言、主题 | 路径、运行和显示参数 |

“用此模型对战”明确切换到自动控制并打开对战页，用户点击开始后才运行。正在对战时不能更换模型。手动编队保留。

语言仍是 **设置 → 界面语言 → 简体中文 / English** 的一个选项，即时生效并保存。切换语言不改变页面、所选关卡、模型或训练参数。

训练运行时收起配置，主按钮随真实状态变为暂停或继续；“停止并保存”仅在有任务时出现。独立后台任务仅提供实际支持的停止功能，界面不会显示虚假的暂停状态。历史后台成绩明确标为“上次后台任务”，新训练开始时清理旧曲线。

## 接线与保留能力

UI 继续通过 Controller / QThread Worker 调用现有 Simulator、MCTS 和 Trainer；未向 UI 复制算法。保留本地关卡库、干员选择、手动／自动／混合控制、机关合法动作、模型比较、日志过滤和 CLI。新增集中导航入口，避免页面合并后旧序号指向错误位置。

关闭应用会等待保存与后台线程清理；关闭期间的检查点事件不会再启动模型扫描。计算互斥立即生效；修改可视化 FPS 会立即到达模拟器控制层。

同一工作区的另一个任务同期完成奖励机制更新。其改动已保留，最终测试包含这些改动；见 [奖励验证说明](STATUS_REWARD_VERIFICATION.md)。本次 UI 工作没有开启正式训练，也不声称模型已学会通关。游戏机制的确认程度仍以 [机制验证说明](STATUS_MECHANICS_VERIFICATION.md) 为准。

## 修改文件

- 工作区与导航：`desktop/main_window.py`、`desktop/widgets/status_bar.py`、`desktop/resources/styles/base.qss`。
- 折叠控件与图表：新增 `desktop/widgets/disclosure.py`，修改 `desktop/widgets/training_chart.py`；整数刻度、单点标记、无数据时隐藏坐标轴。
- 页面：`desktop/views/training_view.py`、`dashboard_view.py`、`simulator_view.py`、`models_view.py`、`settings_view.py`；`game_data_view.py` 增加由主窗口控制的延迟加载入口。
- 验证和启动器：`scripts/verify_desktop_runtime.py`、`scripts/benchmark_desktop.py`、`scripts/build_prts_local_app.py`。
- 新测试：`tests/test_desktop_workspace.py`、`test_desktop_training_simplified.py`、`test_desktop_simulator_simplified.py`、`test_desktop_research_simplified.py`；原 shell 测试适配新导航。
- 使用说明：`desktop/README.md`。

## 验证结果

最终完整回归：**305 passed，0 failed，48.50 秒**。开始时基线为 259 passed。本次新增 32 项 Desktop 测试；同期奖励任务新增 14 项测试，并将一项旧收益标签测试改为剩余收益测试。因此总数净增 46，不能全部算作 UI 新测试。

测试覆盖导航和中英文切换、模型显式交接、运行中保护、配置保存、开始／暂停／继续／停止、旧成绩清理、真实机关操作、线程异常处理和安全关闭。

独立验证目录中的 Qt 控件流程通过，使用 **offscreen** 平台，不等同于原生鼠标操作验收：

- 真实 0-1 回归基线：SUCCESS，11 击杀、0 漏怪，262 条时间轴记录。
- 模型加载及搜索分析：49 个根动作、17 个节点正常显示。
- 小型真实训练：完成 1 轮，145 个样本；策略损失 1.83523、价值损失 0.00860；验证暂停、继续、停止和运行中页面切换。
- 完整状态流：STARTING → RUNNING → PAUSING → PAUSED → RUNNING → STOPPING → IDLE。
- 中英文切换、保存和重新读取一致；验收异常列表为空。

第一次验证使用较大预算，在 60 秒等待上限内未返回，已安全停止并保存，原始记录保留为 `acceptance_first_attempt_timeout.json`。随后使用 4 次搜索模拟的小预算完成流程，没有把超时记录覆盖成成功。

原生 macOS 候选窗口已打开并目视确认简化布局。正式路径的启动器已启动，日志记录服务就绪，进程采样显示原生 Qt 事件循环正常。自动操作工具随后受到历史应用注册信息影响，曾选中旧版备份，未完成对正式窗口的再次连接；因此原生语言切换及完整点击流程仍未独立验收。旧版误开窗口已关闭。

## 应用与性能

- 应用：`dist/Arknights Zero.app`，版本 0.6.0，PySide6 6.11.2。
- `file` 验证主程序为 **Mach-O arm64**；本地签名 strict/deep 验证通过。
- 当前包约 **101 KiB**（103,455 字节），是调用工作区 Python 的本地启动器，必须与本项目及 `.venv` 一起保留，不能把它当成独立可搬移发行包。
- 此次没有重新进行 Nuitka / pyside6-deploy 全量打包。历史独立包信息见 `STATUS_DESKTOP_V01.md`，不适用于当前小启动器。
- offscreen 验证空闲采样：RSS 约 406 MiB，5 秒 CPU 采样 0.2%，MPS 可用；窗口出现约 1.25 秒。它们不是原生冷启动基准。
- 本轮未重新测量 UI 与 CLI 训练开销，未沿用旧版性能数字作为本版结果。

## 数据保存位置与后续

完整记录位于 `research/desktop/simplification/`，最终测试 XML 为 `research/desktop/simplification_final_tests.xml`。`delivery.json` 包含测试增量、架构与签名结果。`runtime/acceptance.json` 保存流程数据；`final_*_offscreen.png` 是布局检查截图，其中训练图使用固定样本数字，不是新的训练成绩。

成功验收检查点：`research/desktop/simplification/workspace/checkpoints/desktop/20260915-075032-010663/latest.pt`。对应样本与指标位于该验证工作区的 `outputs/desktop_training/`；日志保存在其 `logs/`。这些文件与正式训练输出分开保存。

当前没有本次验证遗留的训练任务。下一步优先由实际使用反馈检查日常训练和对战流程，再决定是否增加功能；单独发行包和原生完整自动点击验收作为后续事项。
