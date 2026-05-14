import os
import subprocess
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


GIT_BRANCH, GIT_COMMIT = get_git_info()
