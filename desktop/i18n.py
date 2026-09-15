"""Small runtime translation registry with stable application-facing values.

Views register their English source strings locally. Bind static Qt text with
bind_text; dynamic messages call tr at update time. Combo boxes retain canonical
itemData while only visible itemText changes. No external translation service.
"""

import weakref

from PySide6.QtCore import QObject, QSignalBlocker, Signal, Slot

_TRANSLATIONS = {
    "READY": "就绪",
    "IDLE": "空闲",
    "STARTING": "正在启动",
    "RUNNING": "运行中",
    "PAUSING": "正在暂停",
    "PAUSED": "已暂停",
    "STOPPING": "正在停止",
    "EVALUATING": "评估中",
    "ERROR": "错误",
    "AVAILABLE": "可用",
    "NOT AVAILABLE": "不可用",
    "UNKNOWN": "未知",
    "Unsupported": "暂不支持",
    "Auto": "自动",
    "System": "跟随系统",
    "Light": "浅色",
    "Dark": "深色",
    "None": "无",
}


class LanguageManager(QObject):
    language_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.language = "en"

    def set_language(self, language):
        if language not in ("en", "zh_CN"):
            raise ValueError(f"Unsupported desktop language: {language}")
        if language != self.language:
            self.language = language
            self.language_changed.emit(language)


language_manager = LanguageManager()


def set_language(language):
    language_manager.set_language(language)


def get_language():
    return language_manager.language


def register_translations(entries):
    if any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in entries.items()
    ):
        raise TypeError("Translation entries must map source strings to strings")
    _TRANSLATIONS.update(entries)


def tr(source):
    return _TRANSLATIONS.get(source, source) if get_language() == "zh_CN" else source


class _TextBinding(QObject):
    def __init__(self, target, source, setter, values=None):
        super().__init__(target)
        self._target = weakref.ref(target)
        self.source = source
        self.setter = setter
        self.values = dict(values or {})
        language_manager.language_changed.connect(self.refresh)
        self.refresh()

    @Slot(str)
    def refresh(self, _language=None):
        target = self._target()
        if target is not None:
            translated = tr(self.source)
            getattr(target, self.setter)(
                translated.format(**self.values) if self.values else translated
            )


def bind_text(widget, source, setter="setText", **values):
    """Set and track a static label/button/group/action; returns the widget."""
    bindings = getattr(widget, "_ark_translation_bindings", None)
    if bindings is None:
        bindings = {}
        widget._ark_translation_bindings = bindings
    if setter in bindings:
        bindings[setter].source = source
        bindings[setter].values = dict(values)
        bindings[setter].refresh()
    else:
        bindings[setter] = _TextBinding(widget, source, setter, values)
    return widget


class _ComboBinding(QObject):
    def __init__(self, target, labels):
        super().__init__(target)
        self._target = weakref.ref(target)
        self.labels = dict(labels)
        language_manager.language_changed.connect(self.refresh)
        self.refresh()

    @Slot(str)
    def refresh(self, _language=None):
        target = self._target()
        if target is None:
            return
        blocker = QSignalBlocker(target)
        for index in range(target.count()):
            key = target.itemData(index)
            if key in self.labels:
                target.setItemText(index, tr(self.labels[key]))
        del blocker


def bind_combo(combo, labels):
    """Translate item labels identified by stable itemData, preserving selection."""
    binding = getattr(combo, "_ark_combo_translation_binding", None)
    if binding is None:
        combo._ark_combo_translation_binding = _ComboBinding(combo, labels)
    else:
        binding.labels = dict(labels)
        binding.refresh()
    return combo
