import subprocess


def get_git_info():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        branch, commit = 'unknown', 'unknown'
    return branch, commit


GIT_BRANCH, GIT_COMMIT = get_git_info()
