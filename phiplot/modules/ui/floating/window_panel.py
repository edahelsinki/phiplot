from typing import Any
import panel as pn

class WindowPanel:
    """
    Wrapper for creating a floating `FloatPanel` overlay window.
    Used to display option panels and information on top of the
    main application layout.

    Args:
        name (str): Title of the floating window.
        window_destination (pn.Column): Destination container for rendering the window.
        width (int, optional): Window width in pixels. Defaults to 400.
    """

    def __init__(self, name: str, window_destination: pn.Column, width: int = 400):
        self._name = name
        self._window_destination = window_destination
        self._width = width
        self._contents = []

    @property
    def contents(self) -> list[Any]:
        return self._contents
    
    @contents.setter
    def contents(self, new_contents: list[Any]) -> None:
        """
        Define the contents to display inside the window.

        Args:
            new_contents (list): List of Panel components to include.
        """

        self._contents = new_contents

    def open(self) -> None:
        """
        Open the window by assigning it to the destination container.
        """
        window = self._build()
        window.status = "normalized"
        self._window_destination[:] = [window]

    def _build(self) -> None:
        """
        Construct the floating window with the assigned contents.
        """

        return pn.layout.FloatPanel(
            pn.Column(
                *self.contents,
                sizing_mode="stretch_width"
            ),
            sizing_mode="fixed",
            width=self._width,
            height=None,
            name=self._name,
            status="normalized",
            styles=dict(
                position="fixed",
                top="25%",
                left="50%",
                transform="translateX(-50%)",
                z_index="9999",
            )
        )