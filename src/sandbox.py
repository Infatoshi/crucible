"""Local GPU sandbox for code execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path
from typing import Optional


@dataclass
class SandboxConfig:
    timeout: int = 300
    workdir: Optional[str] = None
    cleanup: bool = True


class Sandbox:
    """Local sandbox with file I/O and command execution."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._workspace: Optional[Path] = None
        self._owns_workspace = False
        self._started = False

    def start(self):
        if self._started:
            return
        if self.config.workdir:
            self._workspace = Path(self.config.workdir).expanduser().resolve()
            self._workspace.mkdir(parents=True, exist_ok=True)
            self._owns_workspace = False
        else:
            self._workspace = Path(tempfile.mkdtemp(prefix="crucible_"))
            self._owns_workspace = True
        self._started = True

    def stop(self):
        if self._owns_workspace and self.config.cleanup and self._workspace:
            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace = None
        self._owns_workspace = False
        self._started = False

    @property
    def workspace_path(self) -> Path:
        if not self._workspace:
            raise RuntimeError("Sandbox not started")
        return self._workspace

    def _resolve_path(self, path: str) -> Path:
        if not self._workspace:
            raise RuntimeError("Sandbox not started")
        if path.startswith("/workspace/"):
            return self._workspace / path[len("/workspace/"):]
        if path.startswith("/workspace"):
            return self._workspace
        if path.startswith("/"):
            return Path(path)
        return self._workspace / path

    def run_command(self, command: str, timeout: Optional[int] = None) -> dict:
        if not self._started:
            raise RuntimeError("Sandbox not started")
        # Remap /workspace paths to actual workspace
        ws = str(self._workspace)
        command = command.replace("/workspace/", f"{ws}/").replace("/workspace", ws)
        # Ensure crucible venv is on PATH so python3/pip find torch, transformers, etc.
        env = dict(os.environ)
        venv_bin = str(Path(__file__).parent.parent / ".venv" / "bin")
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout,
                check=False,
                env=env,
            )
            return {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout or "",
                "stderr": f"Command timed out after {timeout or self.config.timeout}s",
                "returncode": -1,
            }
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1}

    def write_file(self, path: str, content: str) -> bool:
        target = self._resolve_path(path)
        if target == self._workspace or target.is_dir():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return True

    def read_file(self, path: str) -> Optional[str]:
        target = self._resolve_path(path)
        if not target.exists():
            return None
        return target.read_text()

    def file_exists(self, path: str) -> bool:
        if not self._started:
            return False
        return self._resolve_path(path).is_file()

    def list_files(self) -> list[str]:
        if not self._workspace:
            return []
        files = []
        for p in self._workspace.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(self._workspace)))
        return sorted(files)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
