"""Native acceptance of real 3-7 challenge device controls, then safe shutdown."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from PySide6.QtWidgets import QApplication
    from desktop.main_window import MainWindow
    from desktop.widgets.map_picker import MapPicker
    from PySide6.QtCore import QTimer
    app=QApplication([]);window=MainWindow(start_services=False);errors=[]
    window.error_raised.connect(errors.append);window.show();out=ROOT/'research/mechanics/acceptance';out.mkdir(exist_ok=True)
    def until(test,timeout=45):
        end=time.monotonic()+timeout
        while not test() and time.monotonic()<end:
            app.processEvents();time.sleep(.02)
        assert not errors,errors
        assert test(),'Timed out'
    try:
        until(lambda:window.data.snapshot is not None)
        view=window.simulator_view;window.sidebar.setCurrentRow(1)
        def choose():
            picker=next(w for w in app.topLevelWidgets() if isinstance(w,MapPicker))
            picker.search.setText('main_03-07#f#');picker.grab().save(str(out/'map.png'));picker.accept()
        QTimer.singleShot(100,choose);view.map_picker.click()
        assert view.stage.currentData()=='main_03-07#f#'
        view.run_button.click();until(lambda:window.simulator.status=='PAUSED')
        def find(kind):
            for i in range(view.action_combo.count()):
                if json.loads(view.action_combo.itemData(i))['type']==kind:return i
        assert find('PLACE_DEVICE') is not None
        view.action_combo.setCurrentIndex(find('PLACE_DEVICE'));view.apply_button.click()
        until(lambda:'障碍物剩余：2' in window.simulator.last_snapshot.mechanics_status)
        for _ in range(80):
            if find('ACTIVATE_DEVICE') is not None:break
            before=window.simulator.last_snapshot.time;view.step_button.click();until(lambda:window.simulator.last_snapshot.time>before)
        assert find('ACTIVATE_DEVICE') is not None
        view.action_combo.setCurrentIndex(find('ACTIVATE_DEVICE'));view.apply_button.click()
        until(lambda:'反隐剩余' in window.simulator.last_snapshot.mechanics_status)
        window.grab().save(str(out/'devices.png'))
        s=window.simulator.last_snapshot
        (out/'result.json').write_text(json.dumps(dict(stage=s.stage_id,time=s.time,life=s.life,enemies=s.total_enemies,mechanics=s.mechanics_status,status=window.simulator.status,errors=errors),ensure_ascii=False,indent=2))
    finally:
        window.begin_shutdown();until(lambda:window._shutdown_done)

if __name__=='__main__':main()
