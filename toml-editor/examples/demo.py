import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # /// script
    # requires-python = ">=3.9"
    # dependencies = [
    #     "marimo",
    #     "anywidget>=0.9",
    #     "traitlets>=5.0",
    #     "tomli-w>=1.0",
    # ]
    # ///
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 📝 marimo-toml-editor demo

    An interactive TOML config editor widget built with [anywidget](https://anywidget.dev).

    **Features:** tab-per-table, type-aware inputs, rich list/dict editors,
    type badges, search, undo/redo, dirty indicator, keyboard shortcuts.
    """)
    return


@app.cell
def _():
    import sys
    sys.path.insert(0, "../src")  # dev: load from source
    from marimo_toml_editor import TomlConfigEditor

    return (TomlConfigEditor,)


@app.cell
def _(TomlConfigEditor, mo):
    # Instantiate with a sample file path.
    # The widget will create / load the file when you click Open or Save.
    editor = mo.ui.anywidget(TomlConfigEditor(name="demo"))

    # Pre-populate with example data (no file required for the demo)
    editor.widget.data = {
        "title": "My Application",
        "debug": False,
        "version": "1.0.0",
        "primary_color": "#3b82f6",
        "max_retries": 3,
        "tags": ["web", "api", "production"],
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "workers": 4,
        },
        "database": {
            "url": "postgresql://localhost/mydb",
            "pool_size": 10,
            "echo": False,
            "options": {
                "timeout": 30,
                "retries": 3,
            },
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "outputs": ["stdout", "file"],
        },
    }

    editor
    return (editor,)


@app.cell
def _(editor, mo):
    # Reactively display the current data as the user edits
    _data = editor.value.get("data", {})
    mo.md(f"""
    ### Current state

    - **Title:** `{_data.get('title', '—')}`
    - **Debug:** `{_data.get('debug', '—')}`
    - **Tags:** `{_data.get('tags', [])}`
    - **Server port:** `{(_data.get('server') or {}).get('port', '—')}`
    """)
    return


if __name__ == "__main__":
    app.run()
