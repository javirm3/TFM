from __future__ import annotations

from pathlib import Path
import re

import anywidget
import traitlets

from glmhmmt.runtime import load_app_config


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
        button.disabled = !!model.get("disabled");
      };

      button.addEventListener("click", () => {
        if (button.disabled) return;
        const clicks = model.get("clicks") || 0;
        model.set("clicks", clicks + 1);
        model.save_changes();
      });

      model.on("change:label", updateButton);
      model.on("change:disabled", updateButton);

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
    .save-figure-btn:disabled {
      cursor: not-allowed;
      opacity: 0.55;
      background: rgba(229, 231, 235, 0.9);
      border-color: rgba(156, 163, 175, 0.35);
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
      .save-figure-btn:disabled {
        background: rgba(55, 65, 81, 0.8);
        border-color: rgba(107, 114, 128, 0.35);
      }
    }
    """

    clicks = traitlets.Int(0).tag(sync=True)
    label = traitlets.Unicode("Save").tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)


def get_plot_save_format(config_path: Path | None) -> str:
    cfg = load_app_config(config_path)
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


def save_figure(fig, *, results_dir: Path, config_path: Path | None, task_name: str, model_id: str, stem: str) -> Path:
    fmt = get_plot_save_format(config_path)
    out_path = build_plot_path(results_dir, task_name, model_id, stem, fmt)
    if hasattr(fig, "canvas") and fig.canvas is not None:
        fig.canvas.draw()
    save_kwargs = {"bbox_inches": "tight"}
    if fmt != "svg":
        save_kwargs["dpi"] = 300
    fig.savefig(out_path, **save_kwargs)
    return out_path


class PlotSaver:
    def __init__(self, mo, *, results_dir: Path, config_path: Path | None, task_name: str, model_id: str):
        self.mo = mo
        self.results_dir = results_dir
        self.config_path = config_path
        self.task_name = task_name
        self.model_id = model_id
        self.fmt = get_plot_save_format(config_path)
        self._registry: dict[str, dict[str, object]] = {}
        self._save_all = SaveFigureAnyWidget(
            label="Save all model plots",
            disabled=True,
        )
        self._save_all.observe(self._handle_save_all_click, names="clicks")
        self._save_all._save_observer = self._handle_save_all_click
        self._save_all_ui = None

    def _save_one(self, fig, *, stem: str) -> Path:
        return save_figure(
            fig,
            results_dir=self.results_dir,
            config_path=self.config_path,
            task_name=self.task_name,
            model_id=self.model_id,
            stem=stem,
        )

    def _register(self, fig, *, name: str, stem: str) -> None:
        self._registry[stem] = {
            "fig": fig,
            "name": name,
            "stem": stem,
        }
        self._save_all.disabled = not bool(self._registry)

    def _saved_message(self, saved_paths: list[Path]) -> str:
        if not saved_paths:
            return "No files saved."
        if len(saved_paths) == 1:
            return saved_paths[0].name
        return f"{saved_paths[0].name} + {len(saved_paths) - 1} more"

    def save_all(self) -> tuple[list[Path], list[tuple[str, Exception]]]:
        saved_paths: list[Path] = []
        errors: list[tuple[str, Exception]] = []
        for item in list(self._registry.values()):
            try:
                out_path = self._save_one(item["fig"], stem=str(item["stem"]))
                saved_paths.append(out_path)
            except Exception as exc:
                errors.append((str(item["name"]), exc))
        return saved_paths, errors

    def _handle_save_all_click(self, change) -> None:
        if int(change["new"]) <= int(change["old"]):
            return
        if not self._registry:
            self.mo.status.toast(
                "No plots available",
                "<span style='color:#6b7280'>Render the notebook plots first.</span>",
                kind="danger",
            )
            return

        saved_paths, errors = self.save_all()
        if errors:
            _msg = self._saved_message(saved_paths)
            _detail = f"{_msg}; {len(errors)} failed" if saved_paths else f"{len(errors)} failed"
            self.mo.status.toast(
                "Saved with errors" if saved_paths else "Could not save plots",
                f"<span style='color:#6b7280'>{_detail}</span>",
                kind="danger",
            )
            return

        self.mo.status.toast(
            f"Saved {len(saved_paths)} plot{'s' if len(saved_paths) != 1 else ''}",
            f"<span style='color:#6b7280'>{self._saved_message(saved_paths)}</span>",
        )

    def save_all_widget(self, label: str = "Save all model plots"):
        self._save_all.label = label
        if self._save_all_ui is None:
            self._save_all_ui = self.mo.ui.anywidget(self._save_all)
        return self._save_all_ui

    def __call__(self, fig, name: str, *, stem: str | None = None, label: str | None = None):
        _stem = stem or sanitize_stem(name.lower())
        button_label = label or f"Save .{self.fmt}"
        self._register(fig, name=name, stem=_stem)
        widget = SaveFigureAnyWidget(label=button_label)

        def _handle_click(change):
            if int(change["new"]) <= int(change["old"]):
                return
            try:
                out_path = self._save_one(fig, stem=_stem)
                self.mo.status.toast(
                    "Saved",
                    f"<span style='color:#6b7280'>{out_path.name}</span>",
                )
            except Exception as exc:
                self.mo.status.toast(
                    "Could not save figure",
                    f"<span style='color:#6b7280'>{type(exc).__name__}: {exc}</span>",
                    kind="danger",
                )

        widget.observe(_handle_click, names="clicks")
        widget._save_observer = _handle_click
        return self.mo.ui.anywidget(widget)


def make_plot_saver(mo, *, results_dir: Path, config_path: Path | None, task_name: str, model_id: str):
    return PlotSaver(
        mo,
        results_dir=results_dir,
        config_path=config_path,
        task_name=task_name,
        model_id=model_id,
    )


def save_button(
    mo,
    fig,
    *,
    results_dir: Path,
    config_path: Path | None,
    task_name: str,
    model_id: str,
    stem: str,
    label: str = "Save",
):
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
