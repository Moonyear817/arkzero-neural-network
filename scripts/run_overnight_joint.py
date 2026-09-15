"""Bounded, resumable background training of squad selection and battle policy."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import Event,Lock,Thread
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours',type=float,default=6)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/overnight_joint.yaml')
    args=parser.parse_args()
    if not 0<args.hours<=6:parser.error('hours must be in (0, 6]')
    import psutil
    from training.config import load_config
    from training.events import TrainingControl
    from training.trainer import AlphaZeroTrainer
    config=load_config(args.config)
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    lockfile=(out/'run.lock').open('w')
    try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A background training run is already active')
    try:os.nice(8)
    except OSError:pass
    process=psutil.Process()
    started=time.time();deadline=started+args.hours*3600
    record=dict(pid=os.getpid(),process_started=process.create_time(),started_at=datetime.now(timezone.utc).isoformat(),deadline=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),status='STARTING',config=str(args.config),output_dir=str(out),iteration=0,reason='',metrics={})
    mutex=Lock();finished=Event();control=TrainingControl()
    request=out/'stop.request'
    if request.exists():request.unlink()

    def publish(**fields):
        with mutex:
            record.update(fields,updated_at=datetime.now(timezone.utc).isoformat())
            temporary=out/'status.json.tmp'
            temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(out/'status.json')

    def stop(reason):
        publish(status='STOPPING',reason=reason);control.stop()

    def watchdog():
        while not finished.wait(2):
            reason=None
            if request.exists():reason='user_requested'
            elif time.time()>=deadline:reason='six_hour_budget_reached'
            elif shutil.disk_usage(out).free<2*1024**3:reason='low_disk_space'
            elif process.memory_info().rss>2*1024**3:reason='memory_budget_reached'
            battery=psutil.sensors_battery()
            if battery and not battery.power_plugged and battery.percent<15:reason='low_battery'
            if reason:stop(reason);return
            publish(rss_mb=round(process.memory_info().rss/1024**2,1))

    def event(event):
        payload=dict(event.payload)
        if event.kind=='training_metrics':publish(iteration=payload['iteration'],metrics=payload)
        elif event.kind=='episode_finished':
            keys=list(payload.get('selected_squad',()))
            names=[trainer.squad_coordinator.loader.catalog.get(k,{}).get('name',k) for k in keys]
            publish(selected_squad=keys,selected_names=names,last_episode={k:payload.get(k) for k in ('stage','kills','leaks','success','termination','squad_selection_loss')})
        elif event.kind=='checkpoint_saved':
            publish(last_checkpoint=payload['path'])
            archives=sorted(Path(config['checkpoint_dir']).glob('iteration_*.pt'))
            for path in archives[:-3]:path.unlink()
        if event.kind in ('training_metrics','episode_finished','checkpoint_saved','training_finished','error'):
            print(event.kind,json.dumps(payload,ensure_ascii=False,default=str),flush=True)

    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop('signal_requested'))
    publish()
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    thread=Thread(target=watchdog,daemon=True);thread.start()
    try:
        latest=Path(config['checkpoint_dir'])/'latest.pt'
        if latest.exists():config['resume_from']=str(latest)
        trainer=AlphaZeroTrainer(config)
        publish(status='RUNNING',candidate_count=len(trainer.squad_coordinator.keys),squad_size=trainer.squad_coordinator.size)
        summary=trainer.train(callback=event,control=control)
        publish(status='STOPPED' if summary['stopped'] else 'COMPLETE',iteration=summary['iterations_completed'],best_metrics=summary['best_metrics'])
    except Exception:
        publish(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        finished.set();awake.terminate();awake.wait(timeout=5)
        fcntl.flock(lockfile,fcntl.LOCK_UN)


if __name__=='__main__':main()
