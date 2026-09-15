"""Common model/settings actions stay safe when advanced controls are collapsed."""

from dataclasses import replace

import pytest
from PySide6.QtCore import QItemSelectionModel, QObject, Qt, Signal

from desktop.i18n import get_language, set_language
from desktop.services.research_service import CheckpointInfo
from desktop.services.settings_service import SettingsService
from desktop.views.models_view import ModelsView
from desktop.views.settings_view import SettingsView


class ModelStub(QObject):
    models_changed = Signal(object)
    model_loaded = Signal(object)
    evaluation_ready = Signal(object)
    comparison_ready = Signal(object)
    status_changed = Signal(str)

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.busy = False
        self.calls = []

    def refresh(self):
        pass

    def is_busy(self):
        return self.busy

    def select(self, path):
        return next(model for model in self.models if model.path == path)

    def load_model(self, path):
        self.calls.append(("load", path))

    def evaluate(self, path, simulations):
        self.calls.append(("evaluate", path, simulations))

    def compare(self, paths):
        self.calls.append(("compare", paths))

    def stop(self):
        pass


@pytest.fixture(autouse=True)
def isolated_language():
    previous = get_language()
    set_language("en")
    yield
    set_language(previous)


@pytest.fixture
def models_view(qtbot, tmp_path):
    known = CheckpointInfo(
        str(tmp_path / "iteration_000009.pt"),
        "iteration_000009.pt",
        "9",
        1_700_000_000,
        2 * 1024**2,
        metrics=(("success_rate", "0.75"), ("policy_loss", "0.4")),
        latest=True,
    )
    unknown = replace(
        known,
        path=str(tmp_path / "older.pt"),
        name="older.pt",
        iteration="7",
        metrics=(),
        latest=False,
    )
    controller = ModelStub((known, unknown))
    view = ModelsView(controller)
    qtbot.addWidget(view)
    view._models(controller.models)
    return view, controller, known, unknown


def select_path(view, path, *, add=False):
    if not add:
        view.table.clearSelection()
    for row in range(view.table.rowCount()):
        if view.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == path:
            view.table.selectionModel().select(
                view.table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
            return row
    raise AssertionError("Missing model path")


def test_models_summary_preserves_details_and_unknown_evaluation(models_view):
    view, _, known, unknown = models_view
    assert view.table.columnCount() == 4
    assert [view.table.horizontalHeaderItem(i).text() for i in range(4)] == [
        "Model",
        "Iteration",
        "Latest evaluation",
        "Flags",
    ]
    assert view.more.body.isHidden() and view.details_section.body.isHidden()
    row = select_path(view, known.path)
    assert view.table.item(row, 2).text() == "Success 75.0%"
    assert view.table.item(row, 3).text() == "LATEST"
    assert known.path in view.details.toPlainText()
    assert "Size (MB): 2.00" in view.details.toPlainText()
    assert "Created:" in view.details.toPlainText()
    assert "policy_loss: 0.4" in view.details.toPlainText()
    assert "value_loss: UNKNOWN" in view.details.toPlainText()
    row = select_path(view, unknown.path)
    assert view.table.item(row, 2).text() == "UNKNOWN"


def test_battle_action_emits_selected_path_without_loading(models_view, qtbot):
    view, controller, known, unknown = models_view
    select_path(view, known.path)
    with qtbot.waitSignal(view.use_in_simulator) as signal:
        view.use_button.click()
    assert signal.args == [known.path]
    assert controller.calls == []
    select_path(view, unknown.path)
    view.load_button.click()
    view.simulations.setValue(32)
    view.evaluate_button.click()
    assert controller.calls == [("load", unknown.path), ("evaluate", unknown.path, 32)]


def test_computation_gating_keeps_metadata_browsable(models_view):
    view, controller, known, unknown = models_view
    emissions = []
    view.use_in_simulator.connect(emissions.append)
    select_path(view, known.path)
    view.set_compute_available(False)
    assert view.table.isEnabled() and view.refresh_button.isEnabled()
    assert not view.use_button.isEnabled()
    assert not view.load_button.isEnabled() and not view.evaluate_button.isEnabled()
    view._use_in_simulator()
    view._load()
    view._evaluate()
    assert not emissions and not controller.calls
    select_path(view, unknown.path, add=True)
    assert view.compare_button.isEnabled()
    view.compare_button.click()
    assert set(controller.calls[-1][1]) == {known.path, unknown.path}
    view.set_compute_available(True)
    view._use_in_simulator()
    assert not emissions  # Multiple selected rows never choose a model implicitly.
    select_path(view, known.path)
    controller.busy = True
    controller.status_changed.emit("RUNNING")
    assert not view.use_button.isEnabled()
    view._use_in_simulator()
    assert not emissions
    controller.busy = False
    controller.status_changed.emit("READY")
    assert view.use_button.isEnabled()


def test_models_refresh_and_language_keep_selection_and_paths(models_view):
    view, controller, known, unknown = models_view
    select_path(view, known.path)
    controller.models = (unknown, replace(known, metrics=(("success_rate", "1"),)))
    view._models(controller.models)
    assert view._paths() == (known.path,)
    set_language("zh_CN")
    assert view._paths() == (known.path,)
    assert view.use_button.text() == "用此模型对战"
    assert view.evaluate_button.text() == "运行评估"
    assert view.table.horizontalHeaderItem(2).text() == "最近评估"
    row = view.table.selectionModel().selectedRows()[0].row()
    assert view.table.item(row, 2).text() == "通关率 100.0%"
    assert "通关率: 1" in view.details.toPlainText()
    set_language("en")
    assert view._paths() == (known.path,)
    assert view.load_button.text() == "Load model"


def test_settings_primary_controls_visible_advanced_collapsed(qtbot, tmp_path):
    service = SettingsService(tmp_path)
    view = SettingsView(service, {"language": "en"})
    qtbot.addWidget(view)
    view.show()
    assert view.fields["language"].isVisibleTo(view)
    assert view.fields["theme"].isVisibleTo(view)
    assert not view.fields["device"].isVisibleTo(view)
    assert not view.fields["refresh_ms"].isVisibleTo(view)
    view.computation_section.set_expanded(True)
    assert view.fields["device"].isVisibleTo(view)
    assert view.fields["checkpoint_dir"].isVisibleTo(view)
    view.display_section.set_expanded(True)
    assert view.fields["visualization_fps"].isVisibleTo(view)


def test_settings_language_saves_only_language_then_save_keeps_hidden_values(
    qtbot, tmp_path
):
    service = SettingsService(tmp_path)
    original = service.save({"language": "en", "device": "Auto"})
    view = SettingsView(service, original)
    qtbot.addWidget(view)
    view.fields["device"].setCurrentIndex(view.fields["device"].findData("CPU"))
    view.fields["max_log_lines"].setValue(5500)
    view.fields["language"].setCurrentIndex(0)
    assert service.load()["language"] == "zh_CN"
    assert service.load()["device"] == "Auto"
    assert service.load()["max_log_lines"] == original["max_log_lines"]
    assert view.computation_section.toggle.text() == "计算、保存与目录"
    assert view.display_section.toggle.text() == "显示与刷新"
    view.save()
    saved = service.load()
    assert saved["device"] == "CPU" and saved["max_log_lines"] == 5500
    for key, value in original.items():
        if key not in ("language", "device", "max_log_lines"):
            assert saved[key] == value
    set_language("en")
    assert view.save_button.text() == "Settings saved"
