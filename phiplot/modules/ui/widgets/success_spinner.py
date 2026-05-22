import panel as pn
import param

pn.extension()

class SuccessSpinner(pn.viewable.Viewer):
    value = param.Boolean(default=False)
    completed = param.Boolean(default=False)
    status = param.Selector(default="success", objects=["success", "error"])

    def __init__(self, size=40, color="primary", **params):
        super().__init__(**params)
        self._size = size
        self._color = color
        
        self._spinner = pn.widgets.LoadingSpinner(
            size=self._size, color=self._color, value=self.value, 
            width=self._size, height=self._size, margin=5
        )

        self._success_symbol = self._create_icon("M8 12.5L10.5 15L16 9", "#28a745")
        self._error_symbol = self._create_icon("M8 8L16 16M16 8L8 16", "#dc3545")
        
        self._container = pn.Row(self._spinner, align="center")
        self._update_display()

    def _create_icon(self, path, bg_color):
        svg_code = f"""
        <svg viewBox="0 0 24 24" width="{self._size}" height="{self._size}" fill="none" xmlns="http://www.w3.org/2000/svg" style="display: block;">
            <circle cx="12" cy="12" r="10" fill="{bg_color}"/>
            <path d="{path}" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """
        return pn.Row(
            pn.pane.HTML(svg_code, align="center", margin=0),
            width=self._size, height=self._size, margin=5, align="center"
        )

    @param.depends("value", "status", watch=True)
    def _update_display(self):
        self._spinner.value = self.value
        
        if self.value:
            self.completed = True
            self._container[0] = self._spinner
        elif self.completed:
            self._container[0] = self._error_symbol if self.status == "error" else self._success_symbol
        else:
            self._container[0] = self._spinner

    def __panel__(self):
        return self._container

if __name__ == "__main__":
    s = SuccessSpinner(size=50)

    loading_toggle = pn.widgets.Toggle(name="Simulate Loading", width=150)
    loading_toggle.link(s, value="value")

    error_check = pn.widgets.Checkbox(name="Error result")

    @pn.depends(error_check, watch=True)
    def sync_status(is_error):
        s.status = "error" if is_error else "success"

    pn.Column(
        pn.Row(loading_toggle, error_check),
        s
    ).show()