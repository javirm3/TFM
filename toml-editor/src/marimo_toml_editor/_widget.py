"""marimo-toml-editor — Python widget backend."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import anywidget
import traitlets

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import tomli_w
except ImportError:  # pragma: no cover
    tomli_w = None  # type: ignore[assignment]

_STATIC = Path(__file__).parent / "static"


class TomlConfigEditor(anywidget.AnyWidget):
    """Interactive TOML config editor widget for Jupyter and marimo notebooks."""

    # ---- Synced state (Python → JS)
    data: Dict[str, Any] = traitlets.Dict(default_value={}).tag(sync=True)  # type: ignore[assignment]
    name: str = traitlets.Unicode(default_value="config").tag(sync=True)  # type: ignore[assignment]
    status: str = traitlets.Unicode(default_value="").tag(sync=True)  # type: ignore[assignment]
    # toml_text: kept in sync so JS can offer it as a file download
    toml_text: str = traitlets.Unicode(default_value="").tag(sync=True)  # type: ignore[assignment]

    # ---- Command channel (JS → Python)
    command: str = traitlets.Unicode(default_value="").tag(sync=True)  # type: ignore[assignment]
    command_payload: Dict[str, Any] = traitlets.Dict(default_value={}).tag(sync=True)  # type: ignore[assignment]
    command_nonce: int = traitlets.Int(default_value=0).tag(sync=True)  # type: ignore[assignment]

    # ---- Frontend assets
    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    def __init__(self, path: str = "", name: str = "config", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.status = "Ready."
        self.data = {}
        if path:
            self.load(str(Path(path).expanduser()))

    # ------------------------------------------------------------------
    # Keep toml_text in sync whenever data changes
    # ------------------------------------------------------------------

    @traitlets.observe("data")
    def _on_data_change(self, change: Dict[str, Any]) -> None:  # noqa: ARG002
        self._sync_toml_text()

    def _sync_toml_text(self) -> None:
        if tomli_w is None:
            self.toml_text = ""
            return
        try:
            self.toml_text = tomli_w.dumps(self.data)
        except Exception:  # noqa: BLE001
            self.toml_text = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load a TOML file and update the widget state."""
        p = Path(path).expanduser()
        if not p.exists():
            self.data = {}
            self.status = f"File not found: {p}"
            return
        try:
            with p.open("rb") as f:
                obj = tomllib.load(f)
            self.data = obj if isinstance(obj, dict) else {}
            self.name = p.stem
            self.status = f"Loaded: {p.name}"
        except Exception as exc:  # noqa: BLE001
            self.data = {}
            self.status = f"Error loading TOML: {exc}"

    def save(self, path: Optional[str] = None) -> None:
        """Save the current data to a TOML file. Requires tomli-w."""
        if tomli_w is None:
            self.status = "Install tomli-w to enable saving (pip install tomli-w)."
            return
        if not path:
            self.status = "No path specified."
            return
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(tomli_w.dumps(self.data), encoding="utf-8")
            self.status = f"Saved: {p.name}"
        except Exception as exc:  # noqa: BLE001
            self.status = f"Error saving: {exc}"

    # ------------------------------------------------------------------
    # Command handler (JS → Python)
    # ------------------------------------------------------------------

    @traitlets.observe("command_nonce")
    def _on_command(self, change: Dict[str, Any]) -> None:  # noqa: ARG002
        cmd = self.command
        payload = self.command_payload or {}

        if cmd == "load_raw":
            # JS fallback if not Mac native
            content = payload.get("content", "")
            suggested_name = payload.get("name", "")
            try:
                obj = tomllib.load(io.BytesIO(content.encode("utf-8")))
                self.data = obj if isinstance(obj, dict) else {}
                if suggested_name:
                    self.name = Path(suggested_name).stem
                self.status = f"Loaded: {suggested_name or 'file'}"
                # We don't have absolute path, default to current dir for future saves
                self._last_save_path = str(Path.cwd() / (suggested_name or f"{self.name}.toml"))
            except Exception as exc:  # noqa: BLE001
                self.data = {}
                self.status = f"Error parsing TOML: {exc}"

        elif cmd == "mac_native_open":
            import subprocess
            try:
                res = subprocess.run(
                    ['osascript', '-e', 'POSIX path of (choose file with prompt "Select TOML file")'],
                    capture_output=True, text=True, check=False
                )
                path = res.stdout.strip()
                if path:
                    self.load(path)
                    self._last_save_path = path
            except Exception as exc:  # noqa: BLE001
                self.status = f"Dialog error: {exc}"

        elif cmd == "mac_native_save_as":
            import subprocess
            try:
                res = subprocess.run(
                    ['osascript', '-e', f'POSIX path of (choose file name with prompt "Save TOML file as" default name "{self.name}.toml")'],
                    capture_output=True, text=True, check=False
                )
                path = res.stdout.strip()
                if path:
                    content = payload.get("content", "")
                    Path(path).write_text(content, encoding="utf-8")
                    self.name = Path(path).stem
                    self.status = f"Saved: {Path(path).name}"
                    self._last_save_path = path
            except Exception as exc:  # noqa: BLE001
                self.status = f"Dialog error: {exc}"

        elif cmd == "save_local":
            # Save silently to known absolute path, or fallback to name
            content = payload.get("content", "")
            path_str = getattr(self, "_last_save_path", str(Path.cwd() / f"{self.name}.toml"))
            try:
                Path(path_str).write_text(content, encoding="utf-8")
                self.status = f"Saved: {Path(path_str).name}"
            except Exception as exc:  # noqa: BLE001
                self.status = f"Error saving: {exc}"

        else:
            if cmd:
                self.status = f"Unknown command: {cmd}"

        self.command = ""
        self.command_payload = {}

