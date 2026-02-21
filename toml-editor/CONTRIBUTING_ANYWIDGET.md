# Contributing to the anywidget gallery

This document contains everything needed to submit `marimo-toml-editor` to the
[anywidget community gallery](https://anywidget.dev/en/community/).

## PR Target

**Repo:** https://github.com/manzt/built-with-anywidget  
**File to edit:** `assets/manifest.json`

## Manifest entry to add

Add this JSON object to the array in `assets/manifest.json`:

```json
{
  "repo": "javirm3/marimo-toml-editor",
  "description": "Interactive TOML config editor for Jupyter and marimo notebooks. Type-aware inputs, rich list/dict editors, tabs, undo/redo, search, and dark mode.",
  "sourceUrl": "https://raw.githubusercontent.com/javirm3/marimo-toml-editor/main/docs/demo.gif"
}
```

> **Note:** You need to create and upload `docs/demo.gif` (a screen recording of
> the widget in action) before the PR will show a preview image in the gallery.

## Suggested PR title

```
feat: add marimo-toml-editor to gallery
```

## Suggested PR description

```markdown
Hi! I'd like to add **marimo-toml-editor** to the anywidget community gallery.

### What it does

An interactive TOML config editor widget for Jupyter and marimo notebooks.

### Features
- Tab-per-table navigation (each top-level TOML `[table]` is its own tab)
- Type-aware editors: strings, numbers, booleans, hex color picker
- Rich list editor: per-item rows with reorder (↑↓) and delete buttons
- Inline dict editor for shallow nested objects
- Type badges (`str`, `int`, `bool`, `[]`, `{}`) next to each key
- Key search / filter
- Undo / redo history
- Dirty indicator + keyboard shortcuts (Ctrl/Cmd+S save, Ctrl/Cmd+Z undo)
- Raw JSON preview with clipboard copy
- Light + dark mode (follows system preference)

### Install

pip install marimo-toml-editor

### PyPI / GitHub

- GitHub: https://github.com/javirm3/marimo-toml-editor
- PyPI: https://pypi.org/project/marimo-toml-editor
```

## Before submitting

- [ ] Package published to PyPI (`uv build && uv publish`)
- [ ] `docs/demo.gif` created and committed (screen recording of the widget)
- [ ] GitHub repo is public
- [ ] `README.md` has a screenshot / gif at the top
