"""Verify the saved event model through actual desktop controls, then stop."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PySide6.QtWidgets import QApplication
    from desktop.controllers.simulator_controller import SimulatorController
    from desktop.i18n import set_language
    from desktop.views.simulator_view import SimulatorView

    torch.set_num_threads(1)
    app = QApplication([])
    set_language('zh_CN')
    controller = SimulatorController()
    errors, messages = [], []
    controller.error.connect(errors.append)
    controller.log.connect(messages.append)
    view = SimulatorView(controller)
    view.resize(1280, 860)
    view.show()
    destination = ROOT / 'outputs/event_architecture'
    model = destination / 'verification.pt'
    assert Path(view.model_path.text()) == model
    view.mode.setCurrentIndex(view.mode.findData('Hybrid control'))
    view.auto_squad.setChecked(True)

    def until(predicate):
        deadline = time.monotonic() + 45
        while not predicate() and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not errors, errors
        assert predicate(), 'Desktop verification timed out'

    try:
        view.battle_button.click()
        until(lambda: controller.status == 'PAUSED' and controller.last_snapshot is not None)
        worker = controller._worker
        assert len(worker._state.game.squad) == 6
        assert worker._agent.payload['observation_version'] == 3
        assert worker._agent.payload['smoke_only']
        assert '仅完成短训练' in view.model_summary.text()
        first_count = worker._state.decision_count
        view.step_button.click()
        until(lambda: controller.last_snapshot.decision_count > first_count and controller.status == 'PAUSED')
        explanation = worker._agent.last_decision_explanation
        assert explanation and explanation['policy']
        executed = [explanation['selected_action']]
        # Verify an actual placement through the neural desktop path, beyond
        # the first forced wait before DP becomes sufficient.
        for _ in range(20):
            if executed[-1]['type'] == 'DEPLOY':
                break
            count = worker._state.decision_count
            view.step_button.click()
            until(lambda: controller.last_snapshot.decision_count > count and controller.status == 'PAUSED')
            explanation = worker._agent.last_decision_explanation
            executed.append(explanation['selected_action'])
        assert executed[-1]['type'] == 'DEPLOY', executed
        assert any(unit.kind == 'operator' for unit in controller.last_snapshot.units)
        assert any(message.startswith('AI 决策：') for message in messages)
        assert controller.last_snapshot.event_progression
        view.grab().save(str(destination / 'desktop_verification.png'))
        result = dict(status='PAUSED_FOR_VERIFICATION', checkpoint=str(model),
            checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            stage=worker._state.game.stage.id,
            squad=[op.name for op in worker._state.game.squad.values()],
            decision=explanation, executed_actions=executed,
            snapshot_progression=dict(controller.last_snapshot.event_progression),
            model_label=view.model_summary.text(), errors=errors,
            scope='真实桌面加载双网络、自主编队、执行合法等待和AI自动部署；不是策略水平评估。')
    finally:
        assert controller.shutdown(10000)
        until(lambda: not controller.is_busy())
        view.close()
    result['simulation_stopped'] = True
    (destination / 'runtime_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
