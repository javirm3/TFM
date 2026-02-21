# marimo-toml-editor

> An interactive TOML config editor widget for **Jupyter** and **marimo** notebooks — built with [anywidget](https://anywidget.dev).

![demo](docs/demo.png)

## Features

- 🗂 **Tab-per-table navigation** — each top-level TOML table is a tab
- ✏️ **Type-aware editors** — strings, numbers, booleans (toggle), colors (color picker + hex field)
- 📋 **Rich list editor** — per-item rows with reorder (↑↓) and delete
- 🪆 **Inline dict editor** — shallow nested tables rendered as compact key-value rows
- 🏷 **Type badges** — each key shows its type at a glance (`str`, `int`, `bool`, `[]`, `{}`)
- 🔍 **Search / filter** — filter keys within a tab
- ↩ **Undo / Redo** — full history
- 💾 **Save to disk** — requires `tomli-w`
- 🌗 **Light + dark mode** — follows system preference

## Install

```bash
pip install marimo-toml-editor          # read-only
pip install "marimo-toml-editor[save]"  # with save support (tomli-w)
```

## Usage

### In marimo

```python
import marimo as mo
from marimo_toml_editor import TomlConfigEditor

editor = mo.ui.anywidget(TomlConfigEditor("config.toml"))
editor
```

Access the current data reactively:

```python
editor.value["data"]   # dict with the current TOML contents
```

### In Jupyter

```python
from marimo_toml_editor import TomlConfigEditor

w = TomlConfigEditor("config.toml")
w
```

### Start from a dict (no file)

```python
from marimo_toml_editor import TomlConfigEditor

w = TomlConfigEditor()
w.data = {
    "title": "My App",
    "debug": False,
    "server": {"host": "0.0.0.0", "port": 8080},
    "tags": ["web", "api"],
}
```

## API

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `dict` | Current TOML data (synced) |
| `path` | `str` | File path (synced) |
| `name` | `str` | Display name (synced) |
| `status` | `str` | Last operation status message |

| Method | Description |
|--------|-------------|
| `load(path)` | Load a TOML file |
| `save(path?)` | Save to disk (requires `tomli-w`) |

## Development

```bash
git clone https://github.com/javirm3/marimo-toml-editor
cd marimo-toml-editor
uv run marimo edit examples/demo.py
```

## License

MIT
