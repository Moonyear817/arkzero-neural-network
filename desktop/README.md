> 干员养成与技能审计更新：对战“干员养成”、训练高级配置中的同名选项已可用。全员特殊技能与召唤物战斗效果尚未全部实现；准确范围见 [STATUS_PROGRESSION_SKILLS.md](../STATUS_PROGRESSION_SKILLS.md)。

# Arknights Zero Desktop 0.6

macOS 上的 Arknights Zero 观察与控制工作区。界面使用 PySide6，调用已有的
Simulator、Plain UCT、Neural PUCT 和 AlphaZeroTrainer；Core 不导入 Qt，模拟、
搜索与训练的 CLI 入口仍然可以独立运行。

当前界面已简化为三个主要页面：**训练 / Training、对战 / Battle、模型 / Models**。
侧栏下方保留 **研究工具 / Research tools** 和 **设置 / Settings**。高级参数、
详细指标和战斗时间轴可以按需展开，已有能力继续保留在同一个窗口中。

本轮界面改动、验收证据和本机启动器结果见
[STATUS_DESKTOP_SIMPLIFIED.md](../STATUS_DESKTOP_SIMPLIFIED.md)。
奖励机制的独立更新与暂停状态见
[STATUS_REWARD_VERIFICATION.md](../STATUS_REWARD_VERIFICATION.md)。
完整地图库、职业编队与自主编队的范围见
[STATUS_MAPS_JOINT.md](../STATUS_MAPS_JOINT.md)；战斗机制的已知限制见
[STATUS_MECHANICS_VERIFICATION.md](../STATUS_MECHANICS_VERIFICATION.md)。
界面可运行或模型能通关，不代表未知游戏机制已完成验证。

## 启动

在仓库根目录使用项目虚拟环境；需要 Python 3.11 或更高版本：

```sh
.venv/bin/python -m pip install -e '.[alphazero,desktop,desktop-test,test]'
.venv/bin/python scripts/run_desktop.py
```

本机应用入口为 `../Arknights Zero.app`，可以在 Finder 中双击。
它依赖当前仓库和 `.venv`，必须保留在本仓库的 `dist` 目录中。

**设置 → 界面语言 / Settings → Language** 提供 **简体中文 / English**。
切换立即生效并自动保存，不改变关卡、干员、动作或配置中的内部标识。
其他修改通过 **保存设置 / Save settings** 保存。

## 工作区操作

| 页面 | 常用操作 | 展开后保留的功能 |
| --- | --- | --- |
| 训练 / Training | 选择关卡、AI 自主编队或固定编队、训练起点和实验规模；查看迭代、评估通关率与训练耗时 | 高级配置与 YAML、设备和搜索参数、保存检查点、运行评估、更多指标和图表 |
| 对战 / Battle | 地图、控制方式、当前模型与编队来源；开始、暂停、继续和停止对战 | 更多设置、随机种子、诊断模式、重开、可搜索及过滤的战斗时间轴 |
| 模型 / Models | 四列表格：模型、迭代、最近评估、标记；用此模型对战、运行评估 | 加载检查、比较两个模型、刷新、评估模拟次数、完整元数据和评估详情 |
| 研究工具 / Research tools | 搜索分析、游戏数据、日志与诊断三个标签页 | Neural / MCTS 策略表、逐层查看有限数量的搜索节点、数据字段与完整日志查询 |
| 设置 / Settings | 语言和外观 | 计算、保存与目录；显示与刷新 |

### 训练

主要按钮随状态显示 **开始训练 → 暂停 → 继续**，运行期间另有
**停止并保存 / Stop and save**。配置在运行期间收起并锁定，主页面继续展示
当前任务及实时指标。更多指标中的策略损失、价值损失、Replay Buffer、
MCTS 延迟等信息仍可查看。

可以选择从头开始或继续一个检查点。默认 GUI 会话使用独立的
`checkpoints/desktop/<timestamp>` 和 `outputs/desktop_training/<timestamp>`
目录；显式加载的自定义 YAML 保留其配置字段。界面编辑保存到
`configs/gui_session.yaml`，不会改写默认 YAML。

暂停和停止通过 Core 的安全边界执行。训练中的 **保存检查点 / 运行评估**
请求在完成一轮后处理；暂停时可继续训练以到达该边界。停止保留已完成进度，
未完成的 episode 会被丢弃。最新可恢复检查点始终保留，自动归档设置只控制
额外的编号检查点。为评估选择模型不会自动把它作为训练恢复起点。

页面会显示独立后台训练任务的状态。此类任务支持停止并保存，不提供暂停或
恢复按钮；关闭窗口时会说明后台任务仍会继续。应用内训练则在关闭确认后
安全停止。只有查看配置或打开应用不会自动开始新训练。

### 对战

支持 **AI 自动 / AI automatic、手动 / Manual control、人机协同 / Cooperative**。
AI 自动模式由所选模型选择编队和战斗动作；人机协同可以选择模型编队或手动
编队，并在模型行动与手动合法动作之间切换。手动动作会暂停协同模式下的模型。
暂停时的单步操作推进到下一个决策事件。

**用此模型对战 / Use this model in battle** 会把明确选中的检查点交给对战页，
切换为 AI 自动并清除诊断模式；模型在开始对战时由 Worker 加载。已保存的手动
编队继续保留。正在运行或暂停的对战禁止更换本局模型、关卡和编队。

本地地图库和干员选择器可从相关控件打开，手动编队最多 12 名不同干员。
目录中的数据可浏览范围与 Simulator 实际支持的机制不同，未知和未支持字段
保持明确标注。历史黑角 + 玫兰莎回归方案、随机 Agent 和不部署模式保留在
**更多设置 → 诊断模式** 中。可视化速度与显示 FPS 仅影响播放和重绘，
不改变 Core 的模拟时间步长。

地图使用自制基础图形，战斗时间轴支持搜索、事件过滤和暂停自动滚动。
诊断回放与 AlphaZero 训练保持独立。

### 模型、研究工具与日志

模型列表递归读取检查点旁的 JSON 元数据，不在主线程加载大模型权重。
缺少测量结果时显示 **UNKNOWN / 未知**。加载与评估在后台执行；比较展示
现有评估和元数据，不填入虚构指标。不提供删除检查点操作。

搜索分析显示 Neural Policy、MCTS Policy、访问次数、Q 和网络价值，
只保留有限数量的节点供逐层查看。网络价值是有符号估计，不是校准后的通关
概率。当前搜索从所选关卡的首个可部署决策点开始；直接导入任意正在运行的
对战状态仍是后续扩展。

日志与诊断提供分类、搜索、复制和保存。界面日志行数有上限，完整日志保存在
`logs/app.log`、`simulator.log`、`training.log`、`mcts.log`、`errors.log`。
底部状态栏显示训练和对战状态；系统状态菜单显示进程 CPU、内存、MPS 可用性，
并提供日志入口。

## 后台执行与持久化

Controllers 管理可取消的 QThread Workers。Worker 拥有可变的引擎状态，
向界面发送冻结的快照或 DTO；界面不直接修改后台 GameState。训练只发送
episode 和 iteration 摘要，详细战斗事件用于可视化调试模式。图表刷新节流，
可见日志和曲线点数受限，避免界面拖慢计算。神经对战、训练和模型研究之间
有计算互斥控制。

当前开发启动和本地 `.app` 启动器都使用仓库中的配置、数据、日志和检查点。
启动器会将 `ARKNIGHTS_ZERO_WORKSPACE` 设置为仓库根目录。直接运行 Python
入口时可以通过该变量指定其他可写工作区；缺少数据时使用仓库资源作为回退。
应用不会把训练输出写进 `.app` 包内。

## 验证

本次完整自动回归结果为 **305 passed / 0 failed**：从 259 项基线增加了
32 项 Desktop 测试和 14 项奖励测试，另有一项 Core 测试随奖励语义更新。
本轮桌面运行验收使用
**Qt offscreen**，覆盖实际模拟、训练控制、模型加载、搜索、语言切换与安全
停止，并保存截图和 JSON；这不代表本轮重新完成了原生 macOS 窗口验收。

在仓库根目录执行：

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_desktop_runtime.py \
  --workspace research/desktop/simplification/acceptance_workspace \
  --output research/desktop/simplification/runtime
```

验收脚本也支持原生 Qt 窗口，移除 `QT_QPA_PLATFORM=offscreen` 即可。
`acceptance.json` 中的 `qt_platform` 记录实际使用的平台。`--workspace` 将
验收配置和短小训练输出放入独立目录；验收脚本会运行计算任务，不只是打开窗口。

比较无 UI 与有 UI 的训练开销时，在没有其他计算任务的情况下依次运行：

```sh
.venv/bin/python scripts/benchmark_desktop.py --mode cli \
  --output research/desktop/overhead_cli.json
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/benchmark_desktop.py --mode gui \
  --output research/desktop/overhead_gui.json
```

每种模式默认先预热一次，再测量三次。不要把历史机器或旧版界面的性能数字
当作当前版本的重新测量结果。

## 本机 arm64 应用启动器

训练样本与模型存档已分开：正常桌面训练将样本写入工作区的
`训练数据/<任务名>-<标识>/`，模型仍写入 `checkpoints/desktop/<timestamp>/`。
停止训练后可以删除对应的训练数据子文件夹；再次续训会显示提示并重新收集样本，
已保存的模型参数不会丢失。详见 [训练数据清理说明](../训练数据/使用说明.txt)。

当前 `../Arknights Zero.app` 是本地 **arm64 启动器**，由以下脚本构建：

```sh
.venv/bin/python scripts/build_prts_local_app.py
file '../Arknights Zero.app/Contents/MacOS/ArknightsZeroPRTS'
codesign --verify --deep --strict '../Arknights Zero.app'
```

构建需要 macOS 的 `clang` 和 `codesign`。脚本使用 `clang -arch arm64`
编译入口，并进行本机 ad-hoc 签名。入口通过相对位置找到仓库，执行
`.venv/bin/python scripts/run_desktop.py`。程序名中保留的 `PRTS` 是启动器内部
可执行文件名；Finder 显示的应用名为 **Arknights Zero**。

这个入口需要完整仓库、可用的 `.venv`、GameData 和所需模型；不能把 `.app`
单独复制到另一台机器后运行。应用放在 `arkzero` 顶层，项目和运行环境位于同级 `神经网络` 文件夹。迁移后已更新虚拟环境和设置中的绝对目录。当前本机版本不要求 App Store 或 notarization。

历史官方 `pyside6-deploy` / Nuitka 独立打包实验、签名修复过程、当时约 1.082 GB
的包体以及当时的原生运行记录，统一见
[STATUS_DESKTOP_V01.md](../STATUS_DESKTOP_V01.md)。`pysidedeploy.spec` 和相关
历史脚本仍保留，但这些记录不描述当前 0.6 本地启动器，也不能用于声称当前
`.app` 可以脱离仓库独立分发。
