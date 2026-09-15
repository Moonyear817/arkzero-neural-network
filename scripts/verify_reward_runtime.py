"""Load the saved reward model through desktop controls, take one step, stop."""
import json
import hashlib
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

    torch.set_num_threads(2)
    app = QApplication([])
    set_language('zh_CN')
    controller = SimulatorController()
    errors = []
    controller.error.connect(errors.append)
    view = SimulatorView(controller)
    view.resize(1200, 800)
    view.show()
    output = ROOT/'outputs/reward_verification'
    model = output/'initial.pt'
    # This historical verification explicitly selects its own saved model.
    view.model_path.setText(str(model))
    view._refresh_context()
    view.mode.setCurrentIndex(view.mode.findData('Hybrid control'))
    view.auto_squad.setChecked(True)

    def until(predicate):
        deadline = time.monotonic() + 45
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors, errors
        assert predicate(), 'Desktop verification timed out'

    try:
        view.battle_button.click()
        until(lambda: controller.status == 'PAUSED' and controller.last_snapshot is not None)
        worker = controller._worker
        assert len(worker._state.game.squad) == 6
        assert worker._agent.payload['reward_version'] == 2
        assert worker._agent.payload['training_started'] is False
        assert '尚未训练' in view.model_summary.text()
        initial_decisions = worker._state.decision_count
        view.step_button.click()
        until(lambda: controller.last_snapshot.decision_count > initial_decisions and controller.status == 'PAUSED')
        action = worker._agent.search.last_result.selected_action.to_dict()
        view.grab().save(str(output/'desktop_verification.png'))
        result = dict(status='PAUSED_FOR_VERIFICATION', training_started=False,
            checkpoint=str(model), stage=worker._state.game.stage.id,
            checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            squad=[op.name for op in worker._state.game.squad.values()],
            selected_action=action, decisions=worker._state.decision_count,
            model_label=view.model_summary.text(), errors=errors,
            scope='加载自主编队模型并执行一个合法决策；不是通关评估。')
    finally:
        assert controller.shutdown(10000)
        until(lambda: not controller.is_busy())
        view.close()
    result['simulation_stopped'] = True
    (output/'runtime_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    manifest_path = output/'saved_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name in ('runtime_verification.json', 'desktop_verification.png'):
        path = output/name
        manifest['artifacts'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
