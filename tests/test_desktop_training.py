"""GUI adapter lifecycle tests; synthetic trainers are test doubles only."""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from desktop.controllers.training_controller import TrainingController
from desktop.services.config_service import ConfigService
from desktop.views.training_view import TrainingView


def service_for(tmp_path):
    service = ConfigService(tmp_path)
    service.config_dir.mkdir(parents=True)
    service.default_path.write_text(yaml.safe_dump(service.DEFAULTS))
    return service


class ControlledTrainer:
    def __init__(self, config):
        self.config = config
        self.thread = threading.get_ident()

    def train(self, callback, control):
        from training.events import TrainingEvent

        callback(TrainingEvent("training_started", {"device": "cpu"}))
        for iteration in range(10000):
            if not control.checkpoint():
                break
            callback(
                TrainingEvent(
                    "training_metrics",
                    {"iteration": iteration, "policy_loss": 0.5, "value_loss": 0.25},
                )
            )
            for command in control.take_requests():
                callback(
                    TrainingEvent(
                        "checkpoint_saved"
                        if command == "save_checkpoint"
                        else "evaluation_finished",
                        {"path": "test.pt", "iteration": iteration},
                    )
                )
            time.sleep(0.003)
        callback(
            TrainingEvent("training_finished", {"stopped": control.stop_requested})
        )
        return {"stopped": control.stop_requested}


@pytest.fixture
def controller(qtbot, tmp_path):
    value = TrainingController(
        trainer_factory=ControlledTrainer, config_service=service_for(tmp_path)
    )
    yield value
    value.shutdown(5000)
    qtbot.waitUntil(lambda: not value.is_busy, timeout=5000)


def test_config_new_session_preserves_default(tmp_path):
    service = service_for(tmp_path)
    original = service.default_path.read_bytes()
    config = service.load()
    config["mcts_simulations"] = 32
    path = service.save_session(config)
    assert path.name == "gui_session.yaml"
    assert service.load(path)["mcts_simulations"] == 32
    assert service.default_path.read_bytes() == original
    with pytest.raises(ValueError):
        service.save_session(config, service.default_path)
    assert Path(service.prepare_runtime(config)["checkpoint_dir"]).is_absolute()


def test_start_pause_resume_stop_and_metrics(qtbot, controller):
    states, metrics = [], []
    controller.state_changed.connect(states.append)
    controller.metrics_updated.connect(metrics.append)
    assert controller.start(controller.config_service.load())
    qtbot.waitUntil(lambda: bool(metrics), timeout=10000)
    controller.pause()
    qtbot.waitUntil(lambda: controller.state == "PAUSED", timeout=3000)
    assert controller.is_busy
    controller.resume()
    assert controller.state == "RUNNING"
    controller.stop()
    qtbot.waitUntil(lambda: not controller.is_busy, timeout=3000)
    assert controller.state == "IDLE"
    assert {"STARTING", "RUNNING", "PAUSING", "PAUSED", "STOPPING", "IDLE"} <= set(
        states
    )
    assert metrics[-1]["policy_loss"] == 0.5
    with pytest.raises(TypeError):
        metrics[-1]["policy_loss"] = 1


def test_checkpoint_and_evaluation_requests_are_pending_when_paused(qtbot, controller):
    events, logs = [], []
    controller.event_received.connect(events.append)
    controller.log.connect(logs.append)
    controller.start(controller.config_service.load())
    qtbot.waitUntil(lambda: controller.state == "RUNNING", timeout=10000)
    controller.pause()
    qtbot.waitUntil(lambda: controller.state == "PAUSED", timeout=3000)
    controller.save_checkpoint()
    controller.evaluate()
    assert any("pending" in log and "Resume" in log for log in logs)
    assert not any(event.kind == "checkpoint_saved" for event in events)
    controller.resume()
    qtbot.waitUntil(
        lambda: any(event.kind == "checkpoint_saved" for event in events), timeout=3000
    )
    qtbot.waitUntil(
        lambda: any(event.kind == "evaluation_finished" for event in events),
        timeout=3000,
    )


def test_heavy_factory_runs_on_background_thread(qtbot, tmp_path):
    identifiers = []
    main_thread = threading.get_ident()

    def factory(config):
        identifiers.append(threading.get_ident())
        return ControlledTrainer(config)

    controller = TrainingController(
        trainer_factory=factory, config_service=service_for(tmp_path)
    )
    controller.start(controller.config_service.load())
    qtbot.waitUntil(lambda: bool(identifiers), timeout=10000)
    assert identifiers[0] != main_thread
    controller.stop()
    qtbot.waitUntil(lambda: not controller.is_busy, timeout=3000)


def test_worker_exception_captured_with_traceback(qtbot, tmp_path):
    def broken(config):
        raise RuntimeError("test worker error")

    controller = TrainingController(
        trainer_factory=broken, config_service=service_for(tmp_path)
    )
    errors = []
    controller.error.connect(errors.append)
    controller.start(controller.config_service.load())
    qtbot.waitUntil(lambda: not controller.is_busy, timeout=10000)
    assert controller.state == "ERROR"
    assert "test worker error" in errors[0]
    assert "Traceback" in errors[0]


def test_training_view_initialization_and_controls(qtbot, controller):
    view = TrainingView(controller)
    qtbot.addWidget(view)
    assert view.start_button.isEnabled()
    assert not view.stop_button.isEnabled()
    view.set_compute_available(False)
    assert not view.start_button.isEnabled()
    view.set_compute_available(True)
    view.start_button.click()
    qtbot.waitUntil(lambda: controller.state == "RUNNING", timeout=10000)
    assert view.pause_button.isEnabled()
    assert not view.config_group.isEnabled()
    view.pause_button.click()
    qtbot.waitUntil(lambda: controller.state == "PAUSED", timeout=3000)
    assert view.resume_button.isEnabled()
    view.stop_button.click()
    qtbot.waitUntil(lambda: not controller.is_busy, timeout=3000)


def test_real_training_worker_updates_metrics_and_checkpoint(
    qtbot, tmp_path, simple, operator
):
    from arknights_sim.environment import ArknightsEnv
    from training import AlphaZeroTrainer

    service = service_for(tmp_path)
    config = service.load()
    config.update(
        episodes_per_iteration=1,
        mcts_simulations=1,
        training_steps_per_iteration=1,
        evaluation_episodes=1,
        device="cpu",
        max_depth=2,
    )

    def factory(value):
        return AlphaZeroTrainer(
            value,
            env_factory=lambda: ArknightsEnv(
                stage=simple, squad=[operator], horizon=30, max_decisions=128
            ),
        )

    controller = TrainingController(trainer_factory=factory, config_service=service)
    metrics, errors = [], []
    controller.metrics_updated.connect(metrics.append)
    controller.error.connect(errors.append)
    controller.start(config)
    qtbot.waitUntil(lambda: not controller.is_busy, timeout=20000)
    assert not errors
    assert metrics and metrics[-1]["iteration"] == 1
    assert metrics[-1]["parameter_delta"] > 0
    assert (tmp_path / "outputs/desktop_training/checkpoints/latest.pt").is_file()


def test_controller_import_and_construction_do_not_import_torch():
    code = "from PySide6.QtWidgets import QApplication; from desktop.controllers.training_controller import TrainingController; import sys; app=QApplication([]); c=TrainingController(); assert 'torch' not in sys.modules; assert 'training' not in sys.modules"
    result = subprocess.run(
        [os.sys.executable, "-c", code],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_desktop_defaults_create_unique_runs_and_preserve_custom_config(
    qtbot, controller, tmp_path
):
    view = TrainingView(controller)
    qtbot.addWidget(view)
    settings = {
        "config": str(controller.config_service.default_path),
        "device": "CPU",
        "data_dir": str(tmp_path / "data/real"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "auto_save_checkpoint": False,
    }
    view.apply_defaults(settings)
    first, second = view._run_config(), view._run_config()
    assert first["device"] == "cpu"
    assert first["checkpoint_dir"] != second["checkpoint_dir"]
    assert Path(first["checkpoint_dir"]).parent == tmp_path / "checkpoints/desktop"
    assert first["checkpoint_interval"] == first["iterations"] + 1
    custom = controller.config_service.load()
    custom["checkpoint_dir"] = str(tmp_path / "custom_run/checkpoints")
    custom_path = controller.config_service.save_session(
        custom, tmp_path / "configs/custom.yaml"
    )
    view.load_config(custom_path)
    assert view._run_config()["checkpoint_dir"] == custom["checkpoint_dir"]


def test_engine_unavailable_keeps_training_view_usable(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "desktop.services.training_service.find_spec", lambda name: None
    )
    controller = TrainingController(config_service=service_for(tmp_path))
    view = TrainingView(controller)
    qtbot.addWidget(view)
    assert not controller.availability
    from desktop.i18n import tr

    assert view.engine_label.text() == tr("Training Engine: NOT AVAILABLE")
    assert not view.start_button.isEnabled()
    assert not view.evaluate_button.isEnabled()


def test_full_catalog_squad_survives_config_roundtrip(qtbot, controller, tmp_path):
    from desktop.views.training_view import TrainingView
    service = controller.config_service
    config = service.load()
    config['squad'] = ['char_002_amiya', 'char_103_angel', 'char_212_ansel']
    path = service.save_session(config)
    view = TrainingView(controller)
    qtbot.addWidget(view)
    view.load_config(path)
    assert view.collect_config()['squad'] == config['squad']
    view.save_config()
    assert service.load(path)['squad'] == config['squad']
