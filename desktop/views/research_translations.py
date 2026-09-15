"""Presentation-only translations; identifiers and computation settings stay stable."""

import re

from desktop.i18n import register_translations, tr

register_translations(
    {
        "Game Data": "游戏数据",
        "Models": "模型",
        "MCTS": "MCTS",
        "Stages": "关卡",
        "Operators": "干员",
        "Enemies": "敌人",
        "Skills": "技能",
        "Refresh data": "刷新数据",
        "Reading supported GameData…": "正在读取已支持的游戏数据…",
        "Field": "字段",
        "Value": "数值",
        "Refresh": "刷新",
        "Load model": "加载模型",
        "Evaluate model": "评估模型",
        "Compare selected 2": "比较所选的两个模型",
        "Stop": "停止",
        "Sims": "模拟次数",
        "Model": "模型",
        "Iteration": "迭代",
        "Created": "创建时间",
        "Size (MB)": "大小（MB）",
        "Evaluation": "评估分数",
        "Success": "通关率",
        "Policy Loss": "策略损失",
        "Value Loss": "价值损失",
        "Flags": "标记",
        "BEST": "最佳",
        "LATEST": "最新",
        "UNKNOWN": "未知",
        "Unsupported": "尚未支持",
        "Unavailable": "不可用",
        "None": "无",
        "RUNNING": "运行中",
        "STOPPING": "正在停止",
        "ERROR": "错误",
        "STOPPED": "已停止",
        "READY": "就绪",
        "IDLE": "空闲",
        "Neural PUCT": "神经网络 PUCT",
        "Plain UCT": "基础 UCT",
        "Seed": "随机种子",
        "Simulations": "模拟次数",
        "Run search": "开始搜索",
        "Parent": "上一级",
        "Root": "根节点",
        "ROOT": "根节点",
        "Action / Node": "动作 / 节点",
        "Action": "动作",
        "Neural P": "网络 P",
        "MCTS π": "MCTS π",
        "Visits N": "访问次数 N",
        "Q Value": "Q 值",
        "Good": "良好",
        "Neutral": "中性",
        "Poor": "较差",
        "Terminal": "已终止",
        "Non-terminal": "未终止",
        "DEPLOY": "部署",
        "ACTIVATE_SKILL": "开启技能",
        "RETREAT": "撤退",
        "WAIT": "等待",
        "RIGHT": "向右",
        "UP": "向上",
        "Melantha": "玫兰莎",
        "Noir Corne": "黑角",
        "LEFT": "向左",
        "DOWN": "向下",
        "Only the existing loader’s supported mechanics are shown. Unknown fields stay explicit.": "仅展示当前数据加载器已支持的机制。未知字段会明确标注。",
        "Data load failed; see Logs": "数据加载失败，请查看日志",
        "{count} supported records · {directory}": "{count} 条已支持的数据 · {directory}",
        "Checkpoint metadata loads in the background. Select one model or two to compare.": "模型元数据在后台加载。请选择一个模型，或选择两个进行比较。",
        "Model details, evaluation results and basic comparison appear here.": "这里显示模型详情、评估结果与基本比较。",
        " · processing in background": " · 正在后台处理",
        "{count} checkpoints · scan reads sidecars, never training tensors": "{count} 个检查点 · 列表仅读取元数据文件",
        "Loaded {name} · iteration {iteration} · {device}": "已加载 {name} · 迭代 {iteration} · {device}",
        "Parameters": "参数量",
        "Initial-state value": "初始状态价值",
        "Value is an uncalibrated signed estimate, not a success probability.": "价值是未经概率校准的有符号估计，不代表通关概率。",
        "Evaluation complete · seeds {seeds} · {simulations} simulations": "评估完成 · 种子 {seeds} · {simulations} 次模拟",
        "Saved": "已保存",
        "Evaluation used no root noise and temperature 0.": "评估关闭根节点噪声，温度为 0。",
        "Saved evaluation/metadata comparison:": "已保存的评估结果 / 元数据比较：",
        "UNKNOWN means no measured metadata is available; no inference is fabricated.": "“未知”表示尚无实测元数据，不会填入虚构结果。",
        "Network value: unavailable": "网络价值：不可用",
        "Search begins at the first meaningful deployment decision. No preset action sequence.": "从首个可部署的决策状态开始搜索，不使用预设动作序列。",
        "Double-click a child to inspect its next level.\nTop 20 children; at most 2,000 nodes retained.": "双击子节点查看下一层。\n每层显示前 20 项，最多保留 2,000 个节点。",
        "Value is a signed estimate (Good / Neutral / Poor), not a calibrated win probability.": "价值以良好 / 中性 / 较差描述，是有符号估计，并非校准后的胜率。",
        "None selected — choose one on Models": "尚未选择，请在“模型”页选择",
        "Model: {path}": "模型：{path}",
        "Network value: unavailable (plain UCT)": "网络价值：不可用（基础 UCT）",
        "Network value: {value} · {tendency} · {terminal}": "网络价值：{value} · {tendency} · {terminal}",
        "{algorithm}\nTime {time}s · {simulations} simulations · {nodes} nodes · {wall}s\nSelected: {action}": "{algorithm}\n战斗时间 {time} 秒 · {simulations} 次模拟 · {nodes} 个节点 · 用时 {wall} 秒\n选择：{action}",
        "Plain single-agent UCT · tactical rollout": "单玩家基础 UCT · 通用启发式 rollout",
        "evaluation, no root noise": "评估模式，无根节点噪声",
        "Stage ID": "关卡 ID",
        "Source ID": "源数据 ID",
        "Map Size": "地图尺寸",
        "Waves": "波次",
        "Total enemies": "敌人总数",
        "Routes": "路线",
        "Initial DP": "初始费用",
        "Max DP": "费用上限",
        "DP recovery": "费用回复",
        "Life": "生命点数",
        "Deployment limit": "部署上限",
        "Mechanics": "机制",
        "Support": "支持状态",
        "Current simulator assumptions; real-game validation incomplete": "当前采用模拟器假设，真实游戏机制验证尚未完成",
        "Enemy ID": "敌人 ID",
        "Stage": "关卡",
        "HP": "生命值",
        "ATK": "攻击力",
        "DEF": "防御力",
        "Move Speed": "移动速度",
        "Attack Interval": "攻击间隔",
        "Spawn Times": "出生时间",
        "Name": "名称",
        "Internal ID": "内部 ID",
        "Profession": "职业",
        "Position": "部署类型",
        "Block": "阻挡数",
        "DP Cost": "部署费用",
        "Skill": "技能",
        "Scope": "数据范围",
        "Skill ID": "技能 ID",
        "SP Cost": "技能费用",
        "Initial SP": "初始技力",
        "Duration": "持续时间",
        "Recovery": "回复类型",
        "Trigger": "触发类型",
        "Attack multiplier": "攻击倍率",
        "E0 level 1; potential 1; trust 0": "精英 0 等级 1；潜能 1；信赖 0",
        "Level 1": "等级 1",
        "MELEE": "近战位",
        "RANGED": "高台位",
        "ALL": "不限",
        "AUTO_RECOVERY": "自动回复",
        "MANUAL_TRIGGER": "手动触发",
        "success_rate": "通关率",
        "average_leaks": "平均漏怪数",
        "average_kills": "平均击杀数",
        "average_decisions": "平均决策数",
        "value_estimate": "价值估计",
        "mean_value": "平均价值",
        "policy_divergence": "策略差异",
        "inference_latency": "推理延迟",
        "policy_loss": "策略损失",
        "value_loss": "价值损失",
        "evaluation_score": "评估分数",
        "average_decision_time": "平均决策耗时",
        "wall_time": "实际耗时",
        "stage": "关卡",
        "device": "设备",
        "noise": "噪声",
        "temperature": "温度",
        "False": "否",
        "True": "是",
    }
)


def action_text(source):
    source = source.replace("char_208_melan", tr("Melantha")).replace(
        "char_500_noirc", tr("Noir Corne")
    )
    return re.sub(
        r"\b(DEPLOY|ACTIVATE_SKILL|RETREAT|WAIT|RIGHT|UP|LEFT|DOWN|ROOT)\b",
        lambda match: tr(match.group(0)),
        source,
    )


def field_value(value):
    if value.startswith("Unsupported: "):
        return tr("Unsupported") + ": " + value[len("Unsupported: ") :]
    return tr(value)


from PySide6.QtWidgets import QLabel, QPushButton

from desktop.i18n import bind_text, language_manager


class TranslatedLabel(QLabel):
    def __init__(self, source="", parent=None):
        super().__init__(parent)
        self.source = source
        self.values = {}
        language_manager.language_changed.connect(self._refresh)
        self._refresh()

    def set_template(self, source, **values):
        self.source, self.values = source, values
        self._refresh()

    def _refresh(self, *_):
        self.setText(
            tr(self.source).format(
                **{
                    key: tr(value) if isinstance(value, str) else value
                    for key, value in self.values.items()
                }
            )
        )


def label(source=""):
    return TranslatedLabel(source)


def button(source):
    return bind_text(QPushButton(), source)
