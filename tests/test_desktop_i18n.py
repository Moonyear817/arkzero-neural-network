"""Language changes alter presentation, never canonical settings/action values."""

import json

import pytest
import yaml
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QComboBox, QLabel

from desktop.controllers.training_controller import TrainingController
from desktop.i18n import (
    bind_combo,
    bind_text,
    get_language,
    register_translations,
    set_language,
)
from desktop.models.training_state import TrainingEvent
from desktop.services.config_service import ConfigService
from desktop.services.settings_service import SettingsService
from desktop.views.settings_view import SettingsView
from desktop.views.training_view import TrainingView


@pytest.fixture(autouse=True)
def isolated_language():
    previous = get_language()
    set_language("en")
    yield
    set_language(previous)


def test_rebinding_text_is_bounded_and_dynamic_values_retranslate(qtbot):
    register_translations({"Iteration {number}": "第 {number} 轮"})
    label = QLabel()
    qtbot.addWidget(label)
    for number in range(100):
        bind_text(label, "Iteration {number}", number=number)
    assert len(label.findChildren(QObject)) == 1
    assert label.text() == "Iteration 99"
    set_language("zh_CN")
    assert label.text() == "第 99 轮"
    set_language("en")
    assert label.text() == "Iteration 99"


def test_combo_translation_preserves_canonical_data_and_selection(qtbot):
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Auto", "auto")
    combo.addItem("CPU", "cpu")
    bind_combo(combo, {"auto": "Auto", "cpu": "CPU"})
    changes = []
    combo.currentIndexChanged.connect(changes.append)
    set_language("zh_CN")
    assert combo.currentText() == "自动"
    assert combo.currentData() == "auto"
    set_language("en")
    assert combo.currentText() == "Auto"
    assert changes == []


def test_settings_language_selection_is_immediate_and_persistent(qtbot, tmp_path):
    service = SettingsService(tmp_path)
    values = service.load()
    values["language"] = "en"
    view = SettingsView(service, values)
    qtbot.addWidget(view)
    language = view.fields["language"]
    language.setCurrentIndex(language.findData("zh_CN"))
    assert get_language() == "zh_CN"
    assert service.load()["language"] == "zh_CN"
    assert view.save_button.text() == "保存设置"
    assert view.fields["device"].currentText() == "自动"
    assert view.fields["device"].currentData() == "Auto"
    view.save()
    assert service.load()["device"] == "Auto"
    assert service.load()["theme"] == "System"
    language.setCurrentIndex(language.findData("en"))
    assert get_language() == "en"
    assert service.load()["language"] == "en"


def test_legacy_settings_get_chinese_default_and_language_validation(tmp_path):
    service = SettingsService(tmp_path)
    service.path.parent.mkdir(parents=True)
    service.path.write_text(json.dumps({"device": "CPU", "theme": "Dark"}))
    assert service.load()["language"] == "zh_CN"
    with pytest.raises(ValueError):
        service.save({"language": "invalid"})


def test_training_view_translates_live_without_changing_config(qtbot, tmp_path):
    service = ConfigService(tmp_path)
    service.config_dir.mkdir(parents=True)
    service.default_path.write_text(yaml.safe_dump(service.DEFAULTS))
    controller = TrainingController(config_service=service)
    view = TrainingView(controller)
    qtbot.addWidget(view)
    original = view.collect_config()
    assert view.start_button.text() == "START TRAINING"
    set_language("zh_CN")
    assert view.start_button.text() == "开始训练"
    assert view.config_group.title() == "训练配置"
    assert view.fields["device"].currentText() == "自动"
    assert view.collect_config() == original
    view._event(
        TrainingEvent(
            "episode_finished", {"success": False, "kills": 2, "leaks": 9, "episode": 1}
        )
    )
    assert view.metrics["episode_result"].text() == "未通关 · 击杀 2 / 漏怪 9"
    set_language("en")
    assert view.metrics["episode_result"].text() == "Failure · 2 kills / 9 leaks"
    assert view.collect_config() == original
