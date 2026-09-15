"""Desktop research adapters operate on real core APIs away from the GUI thread."""

import json
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from PySide6.QtCore import Qt

from desktop.controllers.data_controller import DataController
from desktop.controllers.mcts_controller import MCTSController
from desktop.controllers.model_controller import ModelController
from desktop.controllers.research_controller import ResearchController
from desktop.services.research_service import (
    PolicyRow,
    inspect_model,
    list_checkpoints,
    load_game_data,
    search_neural_mcts,
)
from desktop.widgets.policy_table import PolicyTable
from desktop.workers.evaluation_worker import ResearchCancelled
from network import PolicyValueNetwork


@pytest.fixture(autouse=True)
def language_scope():
    from desktop.i18n import get_language, set_language

    previous = get_language()
    set_language("en")
    yield
    set_language(previous)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/real"


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    model = PolicyValueNetwork()
    path = tmp_path / "iteration_000007.pt"
    torch.save({"model_state_dict": model.state_dict(), "iteration": 7}, path)
    path.with_suffix(".json").write_text(
        json.dumps({"iteration": 7, "evaluation": {"success_rate": 0.25}})
    )
    return path


def test_model_listing_nested_sidecar_only_and_unknown(tmp_path, monkeypatch):
    nested = tmp_path / "desktop" / "session"
    nested.mkdir(parents=True)
    for path in [tmp_path / "best.pt", nested / "latest.pt"]:
        path.write_bytes(b"not a serialized checkpoint")
    (tmp_path / "best.json").write_text(
        json.dumps({"iteration": 3, "evaluation": {"success_rate": 0.5}})
    )
    (nested / "latest.json").write_text("not valid json")
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Listing must not load weights")
    )
    rows = list_checkpoints(tmp_path)
    assert len(rows) == 2
    best = next(row for row in rows if row.name == "best.pt")
    latest = next(row for row in rows if row.name == "latest.pt")
    assert best.best and best.iteration == "3" and best.metric("success_rate") == "0.5"
    assert latest.latest and latest.metric("value_loss") == "UNKNOWN"
    assert "Metadata unavailable" in latest.metadata_note
    assert all(row.size == len(b"not a serialized checkpoint") for row in rows)


def test_real_data_is_detached_frozen_and_unsupported_explicit():
    data = load_game_data(DATA)
    assert {"Stages", "Operators", "Enemies", "Skills"} <= {
        record.category for record in data.records
    }
    stage = next(
        record
        for record in data.records
        if record.category == "Stages" and record.id == "0-1"
    )
    fields = dict(stage.fields)
    assert fields["Total enemies"] == "11" and fields["Map Size"] == "9 × 6"
    assert stage.stage_map.tiles
    operator = next(record for record in data.records if record.category == "Operators")
    assert dict(operator.fields)["Profession"] != "Unsupported"
    assert len([r for r in data.records if r.category == "Operators"]) == 431
    with pytest.raises(FrozenInstanceError):
        stage.id = "changed"


def test_checkpoint_load_actual_network_preserves_process_rng(checkpoint):
    before = torch.get_rng_state().clone()
    snapshot = inspect_model(checkpoint, DATA)
    expected = sum(v.numel() for v in torch.load(checkpoint, weights_only=True)['model_state_dict'].values())
    assert snapshot.iteration == "7" and snapshot.parameter_count == expected
    assert -1 <= snapshot.value <= 1
    assert torch.equal(before, torch.get_rng_state())


def test_desktop_load_restores_future_information_setting(tmp_path):
    from desktop.services.research_service import load_model
    model = PolicyValueNetwork(future_events_enabled=False)
    path = tmp_path / 'ablation.pt'
    torch.save(dict(model_state_dict=model.state_dict(), network_metadata=model.architecture_config), path)
    loaded, _ = load_model(path)
    assert loaded.future_events_enabled is False


def test_neural_search_reports_real_priors_visits_and_scalar_tree(checkpoint):
    snapshot = search_neural_mcts(DATA, checkpoint, simulations=2)
    assert snapshot.simulations == 2 and snapshot.node_count >= 2
    assert len(snapshot.rows) > 1
    assert sum(row.neural_probability for row in snapshot.rows) == pytest.approx(1)
    assert sum(row.mcts_probability for row in snapshot.rows) == pytest.approx(1)
    assert sum(row.visits for row in snapshot.rows) == 2
    assert snapshot.network_value is not None
    assert len(snapshot.tree) <= 2000
    assert all(len(node.children) <= 20 for node in snapshot.tree)
    assert all(not hasattr(node, "state") for node in snapshot.tree)
    assert all(
        snapshot.tree[c].parent_id == node.id
        for node in snapshot.tree
        for c in node.children
    )


def test_research_worker_cancellation_is_not_error(qtbot):
    controller = ResearchController()
    errors, statuses = [], []
    controller.error.connect(lambda *value: errors.append(value))
    controller.status_changed.connect(statuses.append)

    def operation(check):
        raise ResearchCancelled()

    assert controller._start(
        operation, lambda result: pytest.fail("Cancelled task returned result")
    )
    qtbot.waitUntil(lambda: not controller.is_busy(), timeout=3000)
    assert not errors and controller.status == "STOPPED"
    assert "RUNNING" in statuses


def test_worker_background_errors_have_traceback_and_thread(qtbot):
    controller = ResearchController()
    threads = []

    def operation(check):
        threads.append(threading.get_ident())
        raise RuntimeError("expected research failure")

    with qtbot.waitSignal(controller.error, timeout=3000) as error:
        controller._start(operation, lambda result: None)
    qtbot.waitUntil(lambda: not controller.is_busy(), timeout=3000)
    assert error.args[0] == "expected research failure"
    assert "Traceback" in error.args[1]
    assert threads[0] != threading.get_ident()
    assert controller.status == "ERROR"


def test_data_controller_background_real_load(qtbot):
    controller = DataController(DATA)
    with qtbot.waitSignal(controller.loaded, timeout=5000) as completed:
        assert controller.load()
        assert not controller.load()  # A second parser cannot overlap this one.
    qtbot.waitUntil(lambda: not controller.is_busy())
    assert completed.args[0].directory == str(DATA)
    assert controller.snapshot == completed.args[0]
    assert controller.shutdown()


def test_models_controller_list_select_load_compare(qtbot, checkpoint):
    other = checkpoint.parent / "latest.pt"
    other.write_bytes(checkpoint.read_bytes())
    controller = ModelController(checkpoint.parent, DATA)
    with qtbot.waitSignal(controller.models_changed):
        controller.refresh()
    qtbot.waitUntil(lambda: not controller.is_busy())
    selected = controller.select(checkpoint)
    assert selected.iteration == "7"
    assert len(controller.compare([checkpoint, other])) == 2
    with qtbot.waitSignal(controller.model_loaded, timeout=5000) as loaded:
        controller.load_model()
    qtbot.waitUntil(lambda: not controller.is_busy())
    assert loaded.args[0].name == checkpoint.name
    assert controller.shutdown()


def test_mcts_controller_search_runs_in_worker(qtbot, checkpoint):
    controller = MCTSController(DATA)
    controller.set_model(checkpoint)
    with qtbot.waitSignal(controller.result_ready, timeout=5000) as result:
        controller.search(simulations=2)
    qtbot.waitUntil(lambda: not controller.is_busy())
    assert result.args[0].simulations == 2 and controller.status == "READY"
    assert controller.shutdown()


def test_policy_table_sorts_numbers_and_unavailable(qtbot):
    table = PolicyTable()
    qtbot.addWidget(table)
    table.set_rows(
        (PolicyRow("small", None, 0.1, 2, -0.1), PolicyRow("large", 0.9, 0.9, 10, 0.4))
    )
    table.sortItems(3, Qt.SortOrder.DescendingOrder)
    assert table.item(0, 0).text() == "large"
    table.sortItems(1, Qt.SortOrder.AscendingOrder)
    assert table.item(0, 0).text() == "small"
    assert table.item(0, 1).text() == "Unavailable"


def test_views_initialization_smoke(qtbot, checkpoint):
    from desktop.views.game_data_view import GameDataView
    from desktop.views.mcts_view import MCTSView
    from desktop.views.models_view import ModelsView

    data = DataController(DATA)
    models = ModelController(checkpoint.parent, DATA)
    mcts = MCTSController(DATA)
    views = [GameDataView(data), ModelsView(models), MCTSView(mcts)]
    for view in views:
        qtbot.addWidget(view)
    qtbot.waitUntil(
        lambda: data.snapshot is not None and bool(models.models), timeout=5000
    )
    assert views[0].entries.count() >= 1
    assert views[1].table.rowCount() == 1
    assert views[2].run_button.isEnabled()
    for controller in (data, models, mcts):
        assert controller.shutdown()
    qtbot.waitUntil(lambda: not data.is_busy() and not models.is_busy())


def test_research_language_switch_preserves_ids_and_action_numbers(qtbot):
    from desktop.i18n import set_language
    from desktop.views.game_data_view import GameDataView
    from desktop.views.mcts_view import MCTSView
    from desktop.views.models_view import ModelsView

    data = DataController(DATA)
    models = ModelController(ROOT / "checkpoints", DATA)
    mcts = MCTSController(DATA)
    # Isolate this language test from background parsing/model operations.
    data.load = lambda: None
    models.refresh = lambda: None
    views = [GameDataView(data), ModelsView(models), MCTSView(mcts)]
    for view in views:
        qtbot.addWidget(view)
    views[0].category.setCurrentIndex(1)
    views[2].mode.setCurrentIndex(1)
    rows = (PolicyRow("DEPLOY char_208_melan @ (3, 4) LEFT", 0.4, 0.6, 12, -0.2),)
    views[2].policy_table.set_rows(rows)
    set_language("zh_CN")
    assert views[0].category.currentText() == "干员"
    assert views[0].category.currentData() == "Operators"
    assert views[1].load_button.text() == "加载模型"
    assert views[2].mode.currentData() == "uct"
    assert views[2].mode.currentText() == "基础 UCT"
    assert views[2].run_button.text() == "开始搜索"
    assert "玫兰莎" in views[2].policy_table.item(0, 0).text()
    assert views[2].policy_table.item(0, 3).value == 12
    set_language("en")
    assert views[0].category.currentText() == "Operators"
    assert views[1].load_button.text() == "Load model"
    assert views[2].mode.currentData() == "uct"
    assert "Melantha" in views[2].policy_table.item(0, 0).text()
    assert views[2].policy_table.item(0, 3).value == 12
