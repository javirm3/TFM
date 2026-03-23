from __future__ import annotations

from pathlib import Path
import re
import tomllib

import anywidget
import traitlets


class SaveFigureAnyWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      el.innerHTML = "";

      const wrap = document.createElement("div");
      wrap.className = "save-figure-wrap";

      const button = document.createElement("button");
      button.className = "save-figure-btn";

      const updateButton = () => {
        button.textContent = model.get("label");
      };

      button.addEventListener("click", () => {
        const clicks = model.get("clicks") || 0;
        model.set("clicks", clicks + 1);
        model.save_changes();
      });

      model.on("change:label", updateButton);

      updateButton();

      wrap.appendChild(button);
      el.appendChild(wrap);
    }

    export default { render };
    """
    _css = """
    .save-figure-wrap {
      display: inline-flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 0.35rem;
    }
    .save-figure-btn {
      border: 1px solid rgba(107, 114, 128, 0.35);
      background: rgba(249, 250, 251, 0.95);
      color: #111827;
      border-radius: 8px;
      padding: 0.45rem 0.9rem;
      font-size: 0.9rem;
      line-height: 1;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease;
    }
    .save-figure-btn:hover {
      background: rgba(243, 244, 246, 1);
      border-color: rgba(107, 114, 128, 0.55);
    }
    @media (prefers-color-scheme: dark) {
      .save-figure-btn {
        background: rgba(31, 41, 55, 0.95);
        color: #f3f4f6;
        border-color: rgba(156, 163, 175, 0.35);
      }
      .save-figure-btn:hover {
        background: rgba(55, 65, 81, 1);
        border-color: rgba(209, 213, 219, 0.45);
      }
    }
    """

    clicks = traitlets.Int(0).tag(sync=True)
    label = traitlets.Unicode("Save").tag(sync=True)


def get_plot_save_format(config_path: Path) -> str:
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    fmt = str(cfg.get("plots", {}).get("save_format", "pdf")).lower().strip(". ")
    if fmt not in {"pdf", "svg"}:
        fmt = "pdf"
    return fmt


def sanitize_stem(stem: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "figure"


def build_plot_path(results_dir: Path, task_name: str, model_id: str, stem: str, fmt: str) -> Path:
    out_dir = results_dir / "plots" / task_name / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{sanitize_stem(stem)}.{fmt}"


def save_figure(fig, *, results_dir: Path, config_path: Path, task_name: str, model_id: str, stem: str) -> Path:
    fmt = get_plot_save_format(config_path)
    out_path = build_plot_path(results_dir, task_name, model_id, stem, fmt)
    if hasattr(fig, "canvas") and fig.canvas is not None:
        fig.canvas.draw()
    save_kwargs = {"bbox_inches": "tight"}
    if fmt != "svg":
        save_kwargs["dpi"] = 300
    fig.savefig(out_path, **save_kwargs)
    return out_path


def make_plot_saver(mo, *, results_dir: Path, config_path: Path, task_name: str, model_id: str):
    fmt = get_plot_save_format(config_path)

    def _save_widget(fig, name: str, *, stem: str | None = None, label: str | None = None):
        _stem = stem or sanitize_stem(name.lower())
        button_label = label or f"Save .{fmt}"
        widget = SaveFigureAnyWidget(label=button_label)

        def _handle_click(change):
            if int(change["new"]) <= int(change["old"]):
                return
            try:
                out_path = save_figure(
                    fig,
                    results_dir=results_dir,
                    config_path=config_path,
                    task_name=task_name,
                    model_id=model_id,
                    stem=_stem,
                )
                mo.status.toast(
                    "Saved",
                    f"<span style='color:#6b7280'>{out_path.name}</span>",
                )
            except Exception as exc:
                mo.status.toast(
                    "Could not save figure",
                    f"<span style='color:#6b7280'>{type(exc).__name__}: {exc}</span>",
                    kind="danger",
                )

        widget.observe(_handle_click, names="clicks")
        widget._save_observer = _handle_click
        return mo.ui.anywidget(widget)

    return _save_widget


def save_button(mo, fig, *, results_dir: Path, config_path: Path, task_name: str, model_id: str, stem: str, label: str = "Save"):
    fmt = get_plot_save_format(config_path)
    return make_plot_saver(
        mo,
        results_dir=results_dir,
        config_path=config_path,
        task_name=task_name,
        model_id=model_id,
    )(
        fig,
        name=label,
        stem=stem,
        label=f"{label} .{fmt}",
    )
