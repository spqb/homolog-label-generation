"""The consolidated runner terminates an active stage and its workers."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AnnotationInterruptTest(unittest.TestCase):
    def test_interrupt_stops_stage_and_worker(self):
        for interrupt in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=interrupt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                stage = root / "stage.sh"
                stage.write_text('''#!/usr/bin/env bash
cd "$(dirname "$0")"
trap '' TERM
printf '%s' "$$" > stage.pid
bash -c 'trap "" INT TERM; echo "$BASHPID" > worker.pid; while :; do sleep 1; done' &
wait
''')
                stage.chmod(0o755)
                launcher = root / "launcher.py"
                launcher.write_text(f'''import signal
import sys
sys.path.insert(0, {str(ROOT / 'src/pipeline')!r})
from run_experiments import command
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda received, frame: (_ for _ in ()).throw(SystemExit(128 + received)))
command(['bash', {str(stage)!r}], False)
''')
                proc = subprocess.Popen([sys.executable, str(launcher)], cwd=root,
                                        start_new_session=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while not (root / 'worker.pid').exists():
                        if proc.poll() is not None or time.monotonic() > deadline:
                            self.fail('Stage worker did not start')
                        time.sleep(0.02)
                    stage_pid = int((root / 'stage.pid').read_text())
                    worker_pid = int((root / 'worker.pid').read_text())
                    if interrupt == signal.SIGINT:
                        os.killpg(proc.pid, interrupt)
                    else:
                        proc.send_signal(interrupt)
                    stdout, stderr = proc.communicate(timeout=5)
                    self.assertEqual(proc.returncode, 128 + interrupt, stdout + stderr)
                    for pid in (stage_pid, worker_pid):
                        stat = Path(f'/proc/{pid}/stat')
                        if stat.exists():
                            self.assertEqual(stat.read_text().split(') ', 1)[1].split()[0], 'Z')
                finally:
                    if (root / 'stage.pid').exists():
                        try:
                            os.killpg(int((root / 'stage.pid').read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if proc.poll() is None:
                        proc.kill()
                    proc.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
