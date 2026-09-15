"""Read-only job status plus an explicit cooperative stop request."""
import json
from pathlib import Path
import psutil
from desktop.paths import WORKSPACE_ROOT


def read_background_training(root=None):
    directory=Path(root or WORKSPACE_ROOT)/'outputs/overnight_joint'
    try:
        state=json.loads((directory/'status.json').read_text())
    except (OSError,ValueError):
        return {}
    try:
        process=psutil.Process(state['pid'])
        alive=process.is_running() and abs(process.create_time()-state['process_started'])<0.1
    except (ValueError,KeyError,psutil.Error):
        alive=False
    state['active']=alive and state.get('status') in ('STARTING','RUNNING','STOPPING')
    if not alive and state.get('status') in ('STARTING','RUNNING','STOPPING'):
        state['status']='INTERRUPTED'
    return state


def request_background_stop():
    state=read_background_training()
    if state.get('active'):
        (WORKSPACE_ROOT/'outputs/overnight_joint/stop.request').write_text('Stop and save at the next safe boundary.\n')
