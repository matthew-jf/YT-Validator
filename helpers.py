import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def load_env(required=None):
    load_dotenv()
    missing = [k for k in (required or []) if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    
    
def get_git_info():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        branch = subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        commit = subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        branch, commit = 'unknown', 'unknown'
    return branch, commit


class TaskStore(dict):

    def __init__(self, path='data/tasks.json'):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            self.update(json.loads(self.path.read_text()))

    def save(self):
        serializable = {
            tid: {k: v for k, v in t.items() if k != 'result'}
            for tid, t in self.items()
        }
        self.path.write_text(json.dumps(serializable, indent=2))

