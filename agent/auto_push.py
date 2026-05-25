"""Background git auto-pusher for `dashboard/data/snapshot.json`.

When the agent runs continuously and the dashboard is on GitHub Pages,
we need fresh snapshots reaching the deployed site. This worker does
that without blocking the trading loop:

    1. Sleep `push_every` seconds.
    2. `git add dashboard/data/snapshot.json` (only the snapshot file —
       never sweeps other working-tree changes).
    3. If the file is dirty, `git commit -m "snapshot <timestamp>"` and
       `git push`.
    4. Failures (network down, auth refused) are logged to stderr and
       the loop continues — never crashes the trader.

Assumptions:
- The repo root holds a working git checkout with `origin` pointing at
  the public agora-perp-agent repo and the user's local credentials
  (SSH key or token) already let `git push` succeed.
- We push to the same branch we're on (typically `main`). Switch via
  the operator's `git checkout` before launching if you want a
  different target.

This is intentionally minimal: no rebase logic, no LFS, no branch
forking. If the snapshot pushes fail repeatedly the operator can fix
git state by hand without restarting the trader.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class AutoPusher:
    """Periodically commits + pushes the dashboard snapshot file.

    Args:
        snapshot_path: absolute path to the file we monitor and stage
            (typically `dashboard/data/snapshot.json` in the public
            repo).
        repo_root: directory containing the `.git` of the public repo.
            All git invocations use this as cwd.
        push_every_s: seconds between push attempts. Default 300 (5
            minutes); GH Pages takes ~60s to redeploy, so cadence
            faster than ~120s is wasted bandwidth.
        branch: branch to push (default: current).
        message_prefix: leading text on every commit message; the
            timestamp is appended for human readability.
    """

    def __init__(
        self,
        *,
        snapshot_path: Path,
        repo_root: Path,
        push_every_s: float = 300.0,
        branch: str | None = None,
        message_prefix: str = "snapshot",
    ) -> None:
        self._path = Path(snapshot_path).resolve()
        self._repo = Path(repo_root).resolve()
        self._interval = max(60.0, push_every_s)
        self._branch = branch
        self._prefix = message_prefix
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="auto-pusher"
        )
        self._last_push_at: float | None = None
        self._last_status: str = "idle"

    def start(self) -> None:
        if not (self._repo / ".git").exists():
            print(
                f"[auto-pusher] no .git in {self._repo}; refusing to start.",
                file=sys.stderr,
            )
            return
        self._thread.start()

    def close(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    def status(self) -> str:
        return self._last_status

    # ------------------------------------------------------------ internals

    def _run(self) -> None:
        # First push: wait `interval` before initial commit so the
        # trader has time to accumulate a non-trivial snapshot.
        while not self._stop.wait(self._interval):
            self._push_once()

    def _push_once(self) -> None:
        if not self._path.exists():
            self._last_status = "no snapshot"
            return
        try:
            # Only stage the snapshot, never sweep other working-tree changes.
            self._git("add", str(self._path))
            # Check if there's anything to commit on this path.
            diff = self._git_capture(
                "diff", "--cached", "--quiet", str(self._path)
            )
            if diff.returncode == 0:
                # No staged changes for this file → nothing to commit.
                self._last_status = "nothing to push"
                return
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            self._git(
                "commit", "-q", "-m", f"{self._prefix} {ts}", str(self._path),
            )
            push_args = ["push", "-q", "origin"]
            if self._branch:
                push_args.append(self._branch)
            self._git(*push_args)
            self._last_push_at = time.time()
            self._last_status = f"pushed at {ts}"
        except subprocess.CalledProcessError as e:
            self._last_status = f"git error: {e}"
            print(f"[auto-pusher] {self._last_status}", file=sys.stderr)
        except OSError as e:
            self._last_status = f"os error: {e}"
            print(f"[auto-pusher] {self._last_status}", file=sys.stderr)

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self._repo),
            env=self._env(),
            check=True,
            capture_output=True,
        )

    def _git_capture(self, *args: str) -> subprocess.CompletedProcess:
        # Like `_git` but doesn't raise on non-zero — used for `diff --quiet`
        # which uses exit-code as boolean.
        return subprocess.run(
            ["git", *args],
            cwd=str(self._repo),
            env=self._env(),
            capture_output=True,
        )

    @staticmethod
    def _env() -> dict[str, str]:
        # Inherit shell env so SSH keys / ssh-agent / GIT_SSH_COMMAND
        # carry through. Suppress git's interactive prompts so a missing
        # credential fails fast instead of hanging the worker thread.
        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        return env
