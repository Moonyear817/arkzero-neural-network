# Arknights Zero Desktop — macOS Control Center V0.1

日期：2026-09-15。

**开发模式已完成真实桌面验收：单窗口可以控制 Simulator、加载已有模型、查看神经网络与 MCTS 策略，并启动、暂停、恢复和安全停止现有 AlphaZeroTrainer。设置页支持简体中文 / English，切换立即生效并保存。** 原有 AlphaZero 实验与发布模型保留；桌面产生的新训练使用独立目录。

已生成可在本机启动的 **`dist/Arknights Zero.app`**，主程序架构为 **arm64**。官方部署完成编译后遇到签名参数过长，分批签名修复后通过严格校验。打包应用已实际完成 0-1 回归战斗和一轮 MPS 小型训练。源码验收、性能测量与打包运行证据分别记录，未重复验证的项目明确保留。

## 1. 实测环境

| 项目 | 实际环境 |
|---|---|
| CPU | Apple M2 Pro |
| 物理内存 | 16 GiB |
| 架构 | arm64 |
| 系统 | macOS 26.6.2 |
| Python | 3.12.14 |
| PySide6 / Qt | 6.11.2 |
| PyTorch | 2.14.0 |
| MPS | 可用，后台探测已完成 |
| 本次 UI 训练及对照实验 | CPU，1 个 PyTorch 计算线程 |

用户的目标机器是 M1 / 8GB；本次实际运行机器为 M2 Pro / 16GB。因此以下延迟、内存和性能损失均是这台机器的测量结果，尚未在 M1 / 8GB 上验证。

硬件原始记录：[runtime.json](</Users/yihao/Desktop/arkzero/神经网络/research/alphazero/runtime.json>)。

## 2. UI 架构

```text
PySide6 MainWindow
  ├─ Sidebar + QStackedWidget：八个页面，共用一个工作区
  ├─ Controllers：配置、任务状态、用户命令与结果分发
  ├─ Application Services：调用现有 Core API
  └─ QThread Workers：执行模拟、训练、模型读取、搜索、数据解析
          ↓
  arknights_sim / agents / network / mcts / training
```

主线程负责交互和显示；Simulator、MCTS、训练算法仍在既有 Python Core 中。按钮回调只校验配置、发送命令或启动后台任务。Core 不导入 PySide6，命令行模拟、训练、模型检查与搜索入口继续保留。

界面使用系统调色板、标准 Qt 控件、基础图形和文字。地图由矩形、圆点、线条、部署方向与坐标组成，没有使用官方立绘、图标、贴图或 IPA 美术资源。图表使用 QtCharts，无额外训练可视化框架。

## 3. 新增与扩展文件

| 位置 | 文件及职责 |
|---|---|
| `desktop/` | `app.py`、`main_window.py`、`paths.py`、`i18n.py`：入口、工作区、资源路径与即时语言切换 |
| `desktop/controllers/` | `simulator_controller.py`、`training_controller.py`、`model_controller.py`、`data_controller.py`、`mcts_controller.py`、`system_controller.py`、`research_controller.py` |
| `desktop/workers/` | `simulation_worker.py`、`training_worker.py`、`evaluation_worker.py` |
| `desktop/services/` | `config_service.py`、`settings_service.py`、`training_service.py`、`research_service.py`、`logging_service.py` |
| `desktop/views/` | 八个页面文件，以及 `research_translations.py` |
| `desktop/widgets/` | `metric_card.py`、`training_chart.py`、`stage_map.py`、`battle_timeline.py`、`policy_table.py`、`log_console.py`、`status_bar.py` |
| `desktop/models/` | `ui_state.py`、`training_state.py`、`simulation_state.py` |
| `desktop/resources/` | `styles/base.qss` |
| `scripts/` | `run_desktop.py`、`verify_desktop_runtime.py`、`benchmark_desktop.py`、`finalize_desktop_bundle.py` |
| `configs/` | `desktop_default.yaml`；运行时保存 `gui_session.yaml`、`desktop_settings.json` |
| 测试 | `test_desktop_training.py`、`test_desktop_simulator.py`、`test_desktop_research.py`、`test_desktop_shell.py`、`test_desktop_i18n.py`、`test_desktop_checkpoint_metadata.py` |
| 项目配置 | `pyproject.toml` 增加 Desktop 可选依赖与包发现；`pysidedeploy.spec` 为官方部署配置 |

没有删除或重建既有 Simulator；Desktop 阶段没有更改 Core 数值规则或已有 checkpoint 兼容性。

## 4. 八个页面的可用范围

| 页面 | 已可用 | 当前范围与限制 |
|---|---|---|
| Dashboard / 仪表盘 | Simulator、训练、设备、模型、关卡状态；iteration/episode、loss、成功率、Replay、MCTS、内存与耗时；七组指标曲线 | 未收到真实指标时显示等待/空值，不生成演示数据；模拟速度卡表示当前可视化会话的模拟时间与实际时间比 |
| Simulator / 模拟器 | 真实 0-1、已支持干员、种子；运行/暂停/继续/单步/重启；1×、2×、5×、10×、MAX；地图、路线、单位、阻挡、合法动作与战斗时间线 | 当前核心支持范围为 0-1 与少量干员；包含手动控制和回归基线模式，未扩展复杂技能 |
| Training / 训练 | YAML 配置、Start/Pause/Resume/Stop、保存检查点、评估；实际 iteration/episode、Replay、loss、成功率和耗时 | 控制现有 AlphaZeroTrainer；不在 UI 实现训练算法；安全边界之后才确认暂停或停止 |
| MCTS | Neural P、MCTS 访问分布 π、N、Q 的数字排序；根节点与逐层子节点；真实 Value；Neural PUCT 与 Plain UCT | 从首个有部署动作的状态开始；每层显示前 20 个节点，最多保留 2,000 个 DTO 节点；尚无整棵巨型树画布 |
| Models / 模型 | 递归扫描检查点、BEST/LATEST、日期/大小/元数据、选中、后台加载、实际评估、两个模型基本比较 | 列表只读 sidecar，不反序列化大训练文件；未知指标显示 UNKNOWN；策略散度和独立推理延迟没有元数据时不伪造；不提供删除 |
| Game Data / 游戏数据 | Stages/Operators/Enemies/Skills 分类；真实地图、波次、路线、费用、属性与出生时间 | 已读到 7 条记录；不支持的关卡/机制明确标记；职业等尚未进入当前归一化结构的字段显示 Unsupported |
| Logs / 日志 | ALL/SIM/TRAIN/MCTS/ERROR 分类、搜索、复制、保存；日志持续写盘 | 默认最多显示 10,000 行，旧记录仍保留在磁盘；Core 原始事件 ID、路径和调试细节可保留英文 |
| Settings / 设置 | 简体中文 / English；设备、配置与目录、System/Light/Dark、刷新间隔、日志行数、地图显示 FPS、自动归档选项 | 语言立即生效并自动持久化；路径与默认项按页面说明在重启后采用；最新可恢复、安全停止及最终检查点始终保留 |

语言切换只改变显示文字。下拉选项仍使用稳定的 `itemData`；种子、设备键、动作 ID、搜索模式、数值和训练配置不会随语言变化。研究表中的干员显示使用简短名称，保留原始 Action identity 供 Core 使用。

[中文仪表盘](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/dashboard_chinese.png>) · [中文设置](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/settings_chinese.png>) · [English 设置](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/settings_english.png>)。

## 5. Core 集成与线程边界

`SimulationWorker` 独占自己的 Simulator/Environment 状态。前台接收冻结的 `SimulationSnapshot`、`UnitSnapshot` 和战斗事件 DTO；地图和控件不持有可修改的后台 GameState。手动操作通过合法动作的序列化表示提交，仍由现有 Environment 校验。

`TrainingWorker` 在后台加载 PyTorch 与 `AlphaZeroTrainer`，直接调用 `trainer.train(callback=…, control=…)`。Core 的 `TrainingEvent` 和 `TrainingControl` 不依赖 Qt；UI Adapter 将 iteration、episode、metrics、evaluation、checkpoint 等事件转成 Qt signals。训练模式只发送聚合指标和对局摘要，不逐个推送模拟器物理事件。

`EvaluationWorker` 承载 GameData 解析、检查点读取、模型检查、完整模型评估以及搜索。模型使用 `torch.load(weights_only=True)`；初始化已加载模型时不消耗进程 RNG。搜索调用现有 `PUCTSearch` / `MCTSSearch`，评估调用现有 `play_episode(training=False)`，不建立第二套算法。

后台异常捕获完整 traceback，经信号交给主窗口显示错误，并写入错误日志。研究任务支持协作停止。训练与模型/MCTS 的计算入口互斥，避免同时操作进程级 PyTorch 设置，也避免在小内存机器上叠加多个重计算任务。

## 6. 暂停、停止、关闭和保存

训练状态明确区分 STARTING、RUNNING、PAUSING、PAUSED、STOPPING、IDLE、ERROR。点击暂停后，Core 到达安全边界才确认 PAUSED；继续和停止通过线程安全控制对象发送，不强行终止 QThread。

关闭有运行中训练的窗口会询问是否停止并退出。确认后主窗口保持事件循环运行，等待后台任务安全结束；已完成训练工作由 Core 的原子检查点机制保存。模型加载/搜索/数据任务也先发出停止请求再释放线程。

UI 配置写入新的 `configs/gui_session.yaml`，不直接覆盖默认 YAML。新的桌面训练默认写入 `checkpoints/desktop/<时间戳>/` 与对应的实验输出目录。扫描列表从小型 sidecar 获取指标；训练线程为新 checkpoint 原子写入 sidecar，不让主线程打开含 Replay 的大文件。

已有 AlphaZero 13 轮、65 局 self-play 的实验和发布的 `checkpoints/best.pt` / `latest.pt` 保留。UI 中加载模型会选择该 checkpoint 用于研究/评估；恢复训练仍通过明确的恢复配置进行。

## 7. 真实桌面验收

已运行 `python scripts/run_desktop.py` 并通过原生 macOS Qt 窗口执行真实操作。完整记录：[acceptance.json](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/acceptance.json>)。

| 验收项 | 实测结果 |
|---|---|
| 主窗口出现 | 0.6307 s，源码开发运行 |
| 后台服务准备完成 | 1.8247 s |
| MPS 探测 | Available |
| Simulator 单步 | 时间确实前进 |
| Simulator 完整战斗 | 11 kills、0 leaks、SUCCESS，战斗时间 89.6667 s |
| 战斗时间线 | 262 条实际事件 |
| 模型加载 | `best.pt`，iteration 13 |
| Neural PUCT 页面 | 49 个合法动作、17 个节点；真实 Network Value = −1.0 |
| 训练暂停/恢复 | STARTING → RUNNING → PAUSING → PAUSED → RUNNING |
| 安全停止 | STOPPING → IDLE，保存可恢复 checkpoint |
| 训练中页面导航 | 可用 |
| 新 checkpoint | Models 页面可识别 |
| 中文 / English | 即时切换与保存成功，最终保留简体中文 |
| 异常记录 | 验收过程中 0 条错误 |

训练控制验收实际完成一轮指标：2 个 self-play episodes、290 个状态、Replay 290/1024、16 MCTS simulations；policy loss 0，value/total loss 0.8779209554，参数变化范数 0.6563104987。该小型 GUI 训练的 self-play 与 evaluation 成功率均为 0，评估平均漏怪 11。它验证真实训练与安全控制，不证明这次短训练提升了策略。

安全停止产生的检查点：[latest.pt](</Users/yihao/Desktop/arkzero/神经网络/checkpoints/desktop/20260915-045205-846419/latest.pt>)。

[模拟器运行](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/simulator_running.png>) · [战斗完成](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/simulator_complete.png>) · [训练实时指标](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/training_live.png>) · [中文 MCTS](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/runtime/mcts_chinese.png>)。

另一次实际 Models 按钮评估使用已有 iteration 13 模型、seed 80000、16 simulations，得到 11 kills / 0 leaks / 72 decisions；后台评估 6.03 s 期间，20 ms GUI 定时器触发 294 次。完整决策数据：[模型评估 JSON](</Users/yihao/Desktop/arkzero/神经网络/outputs/desktop/evaluations/best_1789447239107559000.json>)；操作记录：[research_ui_acceptance.json](</Users/yihao/Desktop/arkzero/神经网络/outputs/desktop/research_ui_acceptance.json>)。

已训练模型的 Value 饱和到 −1 是已有 AlphaZero 实验的真实现象。页面将其显示为原始有符号值和“较差”，没有换算为胜率；它与该模型的部分 PUCT 评估能通关同时存在，仍需后续研究。

## 8. 测试

Desktop 开发前的完整 Core 基线为 **193 passed、0 failures、0 errors**。其中包含原有 Simulator、Environment、Plain MCTS 与 AlphaZero 测试，并非只假定最早的 56 项仍然通过。

最终整库测试为 **238 passed、0 failures、0 errors、0 skipped**，即保留 193 项既有测试并新增 45 项 Desktop 测试。

测试覆盖训练控制状态、配置读写、真实 Core Trainer 调用、后台异常、暂停与停止、检查点及 sidecar、Simulator 控制/快照隔离、地图和时间线、模型扫描和加载、实际 PUCT 数据、线程生命周期、数字排序、主窗口初始化，以及中文/English 切换与持久化。

原始结果：[core_baseline.xml](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/core_baseline.xml>) · [test_results.xml](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/test_results.xml>)。

## 9. UI 内存、CPU 和训练性能

原生验收开发进程在服务就绪后的 5 s 短窗口中，空闲 CPU 为 **10.3%**，RSS **410.97 MiB**；完成模拟、模型加载、MCTS 和训练后 RSS **607.81 MiB**。CPU 为 `psutil.Process.cpu_percent()` 的进程指标，不是整机负载或 Apple GPU 使用率；长期稳定空闲值仍未验证。

随后两次更长的可见窗口测量混入了 Simulator 操作：一次日志显示采样期间正在战斗，另一次窗口收到三次 Simulator 启动错误。因此两组补测已标记 **INVALID_CONTAMINATED**，保留原始数据，但不将其 CPU 平均值用作空闲结论，也不据此归因于 QtCharts 或定时器。记录：[idle_telemetry.json](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/idle_telemetry.json>) · [组件探测说明](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/idle_component_probe.md>)。

CLI / GUI 对照使用相同真实 Trainer 配置：CPU、1 线程、seed 54321、1 iteration、2 self-play episodes、8 simulations、2 training steps、batch 8、Replay 容量 1024、2 evaluation episodes。两种模式各运行 1 次预热和 3 次计入结果的重复，包含全部重复数据，没有挑选最快一次。

| 指标 | 无 UI | 打开 UI |
|---|---:|---:|
| 计入结果的平均耗时 | 6.3897 s | 6.7639 s |
| 各次耗时 | 6.5109 / 6.2702 / 6.3881 s | 6.4634 / 6.7333 / 7.0951 s |
| 界面带来的平均耗时增幅 | — | **5.8568%** |
| 各次结束 RSS | 346.70–366.78 MiB | 570.81–580.58 MiB |

对照的 policy loss、value loss、total loss、Replay 大小、evaluation success rate、self-play success rate 六项指标完全一致。GUI 测量期间切换 Dashboard/Training，100 ms 定时器共触发 281 次，最大观测间隔 186.8 ms。

这组短训练负载下的增幅低于用户的 10% 目标；它不代表所有训练参数、MPS 或 M1 / 8GB 的结果。训练时图表按受限刷新频率批量更新，每条曲线最多保留 300 个显示点；详细 Battle Trace 仅用于模拟器调试显示。

原始数据：[对照汇总](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/overhead_comparison.json>) · [CLI 全部重复](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/overhead_cli.json>) · [GUI 全部重复](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/overhead_gui.json>)。

## 10. 日志与错误处理

使用 Python logging 写入 `logs/app.log`、`simulator.log`、`training.log`、`mcts.log`、`errors.log`。显示日志采用有界队列，超过 UI 显示上限不删除已写入磁盘的历史。用户可以按类别和文本检索、复制所选内容或另存日志。

后台错误对话框包含简要错误与完整 traceback。开发中通过测试发现并修复了研究 worker 完成后保留 Python operation closure 的生命周期问题；任务结束会解除该引用，再释放 Qt 线程对象，避免销毁时发生崩溃。

## 11. 打包结果

已配置官方 `pyside6-deploy`，入口为 `scripts/run_desktop.py`，使用 standalone 模式；配置包含 Desktop、现有 Core、GameData、默认配置和现有模型。目标为 `dist/Arknights Zero.app`。没有为了缩小包体移除必要的 PyTorch 依赖。

官方 `pyside6-deploy` 使用 Nuitka 4.2.1，约 25 分钟完成 2,996 个 C 单元编译。最终 codesign 一次传入含 10,129 个 PyTorch header 的路径列表，报 `command line was too long`。部署包装器随后仍复制产物并返回 0，因此该退出码本身不能作为成功证据。

修复脚本 `scripts/finalize_desktop_bundle.py` 将嵌套对象按每批 64 个进行本机 ad-hoc 签名，再签整个 bundle。104 个 Mach-O 文件与 10,129 个 header 文件完成处理；最终 `codesign --verify --deep --strict` 返回 0。未删除依赖或模型，未重新编译已成功的产物。

| 项目 | 结果 |
|---|---|
| 应用 | [Arknights Zero.app](</Users/yihao/Desktop/arkzero/神经网络/dist/Arknights Zero.app>) |
| 文件大小 | **1,082,162,805 bytes**，约 1.082 GB / 1.008 GiB |
| 主程序架构 | `Mach-O 64-bit executable arm64` |
| 本机签名 | ad-hoc；deep / strict 验证通过 |
| 独立启动 | 独立可执行文件与隔离工作目录启动成功；通过 macOS LaunchServices 打开 `.app`，原生窗口正常显示 |
| 打包 Simulator | 0-1 回归基线：**11 kills / 0 leaks / life 20 / WIN**，战斗时间 89.6667 s；地图与时间线可见 |
| 打包 MPS | 可用；完成训练的真实指标中 `device=mps` |
| 打包训练控制 | 实际观察到 STARTING → RUNNING → PAUSING → PAUSED → STOPPING → IDLE，安全保存 iteration 0 检查点 |
| 打包完整小型训练 | 另一次运行完成 iteration 1、2 episodes、290 states，正常结束并保存编号与 latest 检查点 |
| 检查点列表刷新 | 保存后扫描回到 READY，生成对应 sidecar |
| Models 页 Load 按钮 | 源码模式已通过；打包版本未单独重复此项 |
| 首次启动时间 | **未取得有效冷启动测量**；LaunchServices 请求与尚未结束的签名重叠，不使用该时间差；0.6307 s 仅为源码运行结果 |
| 打包 RSS | 交互会话观测约 392 MiB（首次打开后）、740–761 MiB（MPS 工作后）；非受控空闲或峰值测试 |

打包运行记录来自现有原生应用会话与只读日志核对，最终记录期间没有切换页面或中断正在进行的交互。完整小型训练的 policy loss = 0.1541459262、value loss = 0.8776250780、total loss = 1.0317710042、训练耗时 15.7119 s；self-play 和 evaluation 成功率均为 0，平均漏怪 11。它证明打包后的训练链路可以运行，不证明短训练已经学会通关。

打包应用的可写工作目录为 `~/Library/Application Support/Arknights Zero`，与源码实验分开。实验日志、指标和 checkpoint metadata 已复制到研究记录；原始 checkpoint 保留在该工作目录中。

证据：[bundle_verification.json](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/bundle_verification.json>) · [签名记录](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/signing.json>) · [打包日志](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/packaging.log>) · [打包训练指标](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/bundle_runtime/completed_training_metadata.json>) · [打包战斗日志](</Users/yihao/Desktop/arkzero/神经网络/research/desktop/bundle_runtime/simulator.log>)。

本版为本机研究应用，没有 Developer ID 签名、公证或 App Store 发布。源码、测试、实验和部署文件均已保存；本轮开发与实验到此暂停，不自动启动下一轮训练。

## 12. 已知限制与下一步

1. 目标 M1 / 8GB 尚未实机测试。当前已核实主程序 arm64、本机启动、模拟与 MPS 训练；打包版 Models 页 Load、有效冷启动时间和目标机器内存/速度仍待专项验收。
2. 目前仅有 5 秒窗口的空闲 CPU 10.3% 记录。更长补测混入了操作，已排除；下一次应记录控制器活动并确保没有交互，再检查刷新和图表的真实开销。
3. 训练暂停/保存遵循 Core 安全边界，部分请求要等待当前决策、episode 或 iteration 完成；界面显示请求中状态，不承诺任意指令处即时中断。
4. MCTS 树采用有界分层列表；模型比较主要读取已有评估，未知散度/延迟保持 UNKNOWN。后续可增加按所选状态比较模型、固定种子批量评估和详细 inference 指标。
5. Simulator 的 continuous turning、blocking radius、attack windup、same-frame priority 仍未得到完整真实游戏验证。当前明确规则保持 ASSUMED，UI 成功显示或模型能通关不会将其升级为 VERIFIED。
6. 现有 AlphaZero 的 Value 饱和和训练成功率不稳定继续保留在研究结论中。Desktop 是观察和控制层，没有通过预设答案修饰网络或训练指标。
7. 首次启动复制了 `best.pt`，但没有附带旧模型的 `best.json` 元数据，列表可能显示 UNKNOWN；源码模型实际加载已验证。下一版应同步携带元数据。
8. 保存设置后 `data_dir` 可能保留当前 bundle 的绝对路径。移动 `.app` 后如该路径失效，应在设置里更新数据目录；后续改为自动重新定位随包数据。
9. 目前 bundle 使用 Qt 默认图标及非标准的显示名形式 bundle identifier；后续统一应用标识。PyTorch header 和额外依赖导致包体较大，必要依赖没有为缩小体积而删除。

运行开发版：

```bash
python scripts/run_desktop.py
```

UI 的配置、运行状态与研究展示可以继续扩展；Simulator、AlphaZero 训练和 MCTS 始终能够脱离 UI 独立运行。
