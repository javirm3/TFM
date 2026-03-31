from pathlib import Path

import anywidget
import traitlets


class CoefficientEditorWidget(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("coefficient_editor_widget.js")
    _css = Path(__file__).with_name("coefficient_editor_widget.css")

    title = traitlets.Unicode("Coefficient Editor").tag(sync=True)
    subtitle = traitlets.Unicode("").tag(sync=True)
    features = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    weights = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    original_weights = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    slider_min = traitlets.Float(-10.0).tag(sync=True)
    slider_max = traitlets.Float(10.0).tag(sync=True)
    slider_step = traitlets.Float(0.05).tag(sync=True)

