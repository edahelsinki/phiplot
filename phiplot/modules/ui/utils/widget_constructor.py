import logging
from typing import Any
import panel as pn
from scipy.spatial import distance
from phiplot.modules.utils import *

logger = logging.getLogger(__name__)

class WidgetConstructor:
    def __init__(self, default_param_parser: DefaultParamParser):
        self._default_param_parser = default_param_parser
        self._default_style = dict(sizing_mode = "stretch_width")

        self._widget_map =  {
            "str": lambda spec: self._select(spec),
            "int": lambda spec: self._int_input(spec),
            "float": lambda spec: self._float_input(spec),
            "bool": lambda spec: self._bool_input(spec)
        }

        self._two_cols_accordion_objects = []
        self._one_col_accordion_objects = []
        self._widgets = {}
        self._layouts = {}
        self._construct()

    @property
    def widgets(self):
        return self._widgets
    
    @property
    def values(self):
        return {name: w.value for name, w in self._widgets.items()}
    
    def layouts(self, key):
        if key in self._layouts:
            return self._layouts[key]
        
    def _construct(self) -> dict:
        for k, info in self._default_param_parser:
            bool_widgets = []
            other_widgets = []
            url = ""

            for param, spec in info["params"].items():
                dtype = spec["dtype"]
                if dtype not in self._widget_map:
                    continue
                spec["name"] = param
                spec = self._resolve_spec(spec)
                w = self._widget_map[dtype](spec)
                if dtype == "bool":
                    bool_widgets.append(w)
                else:
                    other_widgets.append(w)
                self._widgets[f"{k}_{param}_{dtype}"] = w

            two_cols_layout = pn.Column(
                pn.Row(
                    pn.Column(*other_widgets, sizing_mode="stretch_width"),
                    pn.Column(
                        pn.pane.Markdown(f"## 📄<a href='{url}' target='_blank'>Docs</a>"),
                        *bool_widgets,
                        sizing_mode="stretch_width"),
                    sizing_mode="stretch_width"
                )
            )

            one_col_layout = pn.Column(
                pn.Column(*(other_widgets + bool_widgets), sizing_mode="stretch_width"),
                pn.pane.Markdown(f"## 📄<a href='{url}' target='_blank'>Docs</a>")
            )
            
            self._layouts[k] = dict(
                one_col = one_col_layout,
                two_cols = two_cols_layout
            )
        
    def _select(self, spec: dict[str, Any]) -> pn.widgets.Select:
        return pn.widgets.Select(
            name = spec.get("name", ""),
            options = spec.get("options", []),
            value = spec.get("value", None),
            **spec.get("style", self._default_style)
        )
    
    def _int_input(self, spec: dict[str, Any]) -> pn.widgets.IntInput:
        return pn.widgets.IntInput(
            name = spec.get("name", ""),
            start = spec.get("start", None),
            end = spec.get("end", None),
            value = spec.get("value", 0),
            **spec.get("style", self._default_style)
        )
    
    def _float_input(self, spec: dict[str, Any]) -> pn.widgets.FloatInput:
        return pn.widgets.FloatInput(
            name = spec.get("name", ""),
            start = spec.get("start", None),
            end = spec.get("end", None),
            value = spec.get("value", 0),
            format="0.0[0000]",
            **spec.get("style", self._default_style)
        )
    
    def _bool_input(self, spec: dict[str, Any]) -> pn.widgets.Checkbox:
        return pn.widgets.Checkbox(
            name = spec.get("name", ""),
            value = spec.get("default", False),
            **spec.get("style", self._default_style)
        )

    def _resolve_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        resolved = spec.copy()
        has_restrictions = resolved.pop("restrictions", None) != None
        resolved["value"] = resolved.pop("default", None)
        
        if has_restrictions:
            if spec["dtype"] == "str":
                if spec["restrictions"] is not None:
                    resolved["options"] = spec["restrictions"]   
            elif spec["dtype"] in ["int", "float"]:
                if spec["restrictions"] is not None:
                    resolved ["start"] = spec["restrictions"][0]
                    resolved ["end"] = spec["restrictions"][1]
        elif spec["name"] == "metric":
            resolved ["options"] = list(distance._METRICS.keys())
        
        return resolved