"""Native UI acceptance for catalog, squad picker and manual deployments."""
import json
import sys
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    torch.set_num_threads(2)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from desktop.main_window import MainWindow
    from desktop.widgets.squad_picker import SquadPicker
    from desktop.widgets.map_picker import MapPicker
    app = QApplication([])
    window = MainWindow(start_services=False)
    errors = []
    window.error_raised.connect(errors.append)
    window.show()
    out = ROOT / 'research/prts_runtime'
    out.mkdir(parents=True, exist_ok=True)

    def until(predicate):
        deadline = monotonic() + 60
        while not predicate() and monotonic() < deadline and not errors:
            app.processEvents()
            sleep(0.02)
        assert not errors, errors
        assert predicate(), 'Timed out'

    try:
        until(lambda: window.data.snapshot is not None)
        records = window.data.snapshot.records
        assert len([r for r in records if r.category == 'Operators']) == 431
        window.sidebar.setCurrentRow(8)
        roster = window.roster
        roster.set_squad(())
        roster.tabs.setCurrentIndex(4)
        roster.search.setText('能天使')
        QTest.qWait(100)
        QTest.mouseClick(roster.checks['char_103_angel'], Qt.MouseButton.LeftButton)
        assert roster.squad() == ('char_103_angel',)
        roster.search.clear()
        QTest.qWait(100)
        window.grab().save(str(out / 'professions.png'))
        maps = MapPicker(ROOT / 'data/real', window)
        maps.show()
        assert maps.entries.count() == 4694
        maps.search.setText('main_00-08')
        QTest.qWait(100)
        assert '生命' in maps.details.toPlainText()
        maps.grab().save(str(out / 'maps.png'))
        maps.close()
        window.sidebar.setCurrentRow(5)
        data = window.data_view
        data.category.setCurrentIndex(data.category.findData('Operators'))
        data.search.setText('阿米娅')
        until(lambda: data.entries.count() >= 3)
        window.grab().save(str(out / 'catalog.png'))
        window.sidebar.setCurrentRow(1)
        view = window.simulator_view
        assert view.mode.count() == 3
        selected = {'char_002_amiya', 'char_103_angel', 'char_212_ansel', 'char_500_noirc'}

        def choose():
            picker = next(w for w in app.topLevelWidgets() if isinstance(w, SquadPicker))
            picker.selected = selected
            picker._filter()
            picker.grab().save(str(out / 'squad.png'))
            picker.accept()

        QTimer.singleShot(100, choose)
        view.pick_squad.click()
        assert view.mode.currentData() == 'Manual control'
        view.run_button.click()
        until(lambda: window.simulator.status == 'PAUSED' and window.simulator.last_snapshot is not None)
        def find_deploy():
            for i in range(view.action_combo.count()):
                p = json.loads(view.action_combo.itemData(i))
                if p['type'] == 'DEPLOY' and p.get('operator_id') == 'char_103_angel':
                    return i
            return None
        for _ in range(20):
            if find_deploy() is not None:
                break
            old = window.simulator.last_snapshot.time
            view.step_button.click()
            until(lambda: window.simulator.last_snapshot.time > old)
        index = find_deploy()
        assert index is not None
        view.action_combo.setCurrentIndex(index)
        view.apply_button.click()
        until(lambda: any(u.id == 'char_103_angel' for u in window.simulator.last_snapshot.units))
        window.grab().save(str(out / 'manual.png'))
        window.simulator.shutdown(5000)
        until(lambda: not window.simulator.is_busy())
        view.model_path.setText(str(ROOT / 'outputs/joint_commissioning/checkpoints/best.pt'))
        view.mode.setCurrentIndex(view.mode.findData('Automatic control'))
        view.speed.setCurrentIndex(view.speed.count()-1)
        deployed = set()
        window.simulator.snapshot.connect(lambda s: deployed.update(u.id for u in s.units if u.id.startswith('char_')))
        print('automatic_ready', view.run_button.isEnabled(), window.simulator.status, view.mode.currentData(), flush=True)
        view.run_button.click()
        print('automatic_started', window.simulator.status, window.simulator.is_busy(), flush=True)
        until(lambda: window.simulator.status == 'FINISHED')
        assert deployed, 'Neural policy never deployed an operator'
        window.grab().save(str(out / 'automatic.png'))
        (out / 'result.json').write_text(json.dumps({'operators':431, 'map_variants':4694, 'modes':3, 'checkbox_click':True, 'squad':sorted(selected), 'manual_deployed':'char_103_angel', 'automatic_deployed':sorted(deployed), 'status':window.simulator.status, 'errors':errors},ensure_ascii=False,indent=2))
    finally:
        window.simulator.shutdown(5000)
        window.begin_shutdown()
        until(lambda: window._shutdown_done)


if __name__ == '__main__':
    main()
