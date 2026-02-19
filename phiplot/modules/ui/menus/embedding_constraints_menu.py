from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.views.embedding_view import EmbeddingView

logger = logging.getLogger(__name__)


class EmbeddingConstraintsMenu(BaseMenu):
    def __init__(self, parent_view: EmbeddingView):
        super().__init__(
            parent_view=parent_view,
            name="Constraints",
            icon="pin"
        )

        self._embedding_data_handler = parent_view.embedding_data_handler

        self._floating_panels = dict(
           add_cp = self._build_add_control_point_panel(),
           add_ml = self._build_add_must_link_panel(),
           add_cl = self._build_add_cannot_link_panel()
        )

        self.menu_items = [
            ("Add Control Point", "add_cp"),
            ("Clear Control Points", "clear_cp"),
            None,
            ("Add Must-Link", "add_ml"),
            ("Clear Must-Links", "clear_ml"),
            None,
            ("Add Cannot-Link", "add_cl"),
            ("Clear Cannot-Links", "clear_cl"),
            None,
            ("Clear All Constraints", "clear_all"),
        ]

        self.callbacks = dict(
            add_cp = lambda: self._floating_panels["add_cp"].open(),
            clear_cp = lambda: self._clear_control_points(),
            add_ml = lambda: self._floating_panels["add_ml"].open(),
            clear_ml = lambda: self._clear_must_links(),
            add_cl = lambda: self._floating_panels["add_cl"].open(),
            clear_cl = lambda: self._clear_cannot_links(),
            clear_all = lambda: self._clear_constraints()
        )

        self.adding_constraint = False

    def remove_control_point(self, event=None) -> None:
        """
        Callback for removing control points from the display.

        Args:
            event: Panel value-change event from `DisplayPanel`.
        """

        if self.adding_constraint:
            return

        old_idx = set(str(s.split(":")[0]) for s in event.old)
        new_idx = set(str(s.split(":")[0]) for s in event.new)
        deleted = list(old_idx - new_idx)

        if deleted:
            formatted = self._embedding_data_handler.remove_control_points(deleted)
            self._parent_view.control_points = formatted

            if self._embedding_handler.must_reembed:
                self._parent_view.embed()
            else:
                self._parent_view.plot.update()

    def remove_must_link(self, event=None) -> None:
        """
        Callback for removing must-links from the display.

        Args:
            event: Panel value-change event from `DisplayPanel`.
        """

        if self.adding_constraint:
            return

        deleted = self._deleted_links(event)
        if deleted:
            formatted = self._embedding_data_handler.remove_must_link(deleted)
            self._parent_view.must_links = formatted
            self._parent_view.plot.update()

    def remove_cannot_link(self, event=None) -> None:
        """
        Callback for removing cannot-links from the display.

        Args:
            event: Panel value-change event from `DisplayPanel`.
        """

        if self.adding_constraint:
            return

        deleted = self._deleted_links(event)
        if deleted:
            formatted = self._embedding_data_handler.remove_cannot_link(deleted)
            self._parent_view.cannot_links = formatted
            self._parent_view.plot.update()

    def _build_add_control_point_panel(self) -> WindowPanel:
        """
        Construct and return a floating window for adding a new control point.

        Returns:
            WindowPanel: Configured panel with index/x/y inputs and submit button.
        """

        window = WindowPanel("Add Control Point", self._window_destination)

        self._widgets["control_point_idx_str"] = pn.widgets.TextInput(
            name="Index", sizing_mode="stretch_width"
        )

        self._widgets["control_point_x_float"] = pn.widgets.FloatInput(
            name="x", value=0, step=0.1, sizing_mode="stretch_width"
        )

        self._widgets["control_point_y_float"] = pn.widgets.FloatInput(
            name="y", value=0, step=0.1, sizing_mode="stretch_width"
        )

        self._widgets["add_control_point_button"] = pn.widgets.Button(
            name="Add Control", **self._styling.default_button_style
        )
        self._widgets["add_control_point_button"].on_click(self._add_control_point)

        window.contents = [
            self._widgets["control_point_idx_str"],
            pn.Row(
                self._widgets["control_point_x_float"],
                self._widgets["control_point_y_float"],
            ),
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["add_control_point_button"],
        ]
        return window
    
    def _build_add_must_link_panel(self) -> WindowPanel:
        """
        Construct and return a floating window for adding a new must-link.

        Returns:
            WindowPanel: Configured panel with two index inputs and submit button.
        """

        window = WindowPanel("Add Must-Link", self._window_destination)

        self._widgets["must_link_idx_1_str"] = pn.widgets.TextInput(name="Index 1", sizing_mode="stretch_width")

        self._widgets["must_link_idx_2_str"] = pn.widgets.TextInput(name="Index 2", sizing_mode="stretch_width")

        self._widgets["add_must_link_button"]  = pn.widgets.Button(
            name="Add Must-Link", **self._styling.default_button_style
        )
        self._widgets["add_must_link_button"].on_click(self._add_must_link)

        window.contents = [
            self._widgets["must_link_idx_1_str"],
            self._widgets["must_link_idx_2_str"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["add_must_link_button"]
        ]
        return window
    
    def _build_add_cannot_link_panel(self) -> WindowPanel:
        """
        Construct and return a floating window for adding a new cannot-link.

        Returns:
            WindowPanel: Configured panel with two index inputs and submit button.
        """

        window = WindowPanel("Add Cannot-Link", self._window_destination)

        self._widgets["cannot_link_idx_1_str"] = pn.widgets.TextInput(name="Index 1", sizing_mode="stretch_width")

        self._widgets["cannot_link_idx_2_str"] = pn.widgets.TextInput(name="Index 2", sizing_mode="stretch_width")

        self._widgets["add_cannot_link_button"] = pn.widgets.Button(
            name="Add Cannot-Link", **self._styling.default_button_style
        )
        self._widgets["add_cannot_link_button"].on_click(self._add_cannot_link)

        window.contents = [
            self.widgets["cannot_link_idx_1_str"],
            self.widgets["cannot_link_idx_2_str"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self.widgets["add_cannot_link_button"]
        ]
        return window

    def _add_control_point(self, event=None) -> None:
        """
        Callback for adding a new control point.

        Args:
            event: Panel button click event.
        """

        idx = self._widgets["control_point_idx_str"].value
        x = self._widgets["control_point_x_float"].value
        y = self._widgets["control_point_y_float"].value

        formatted = self._embedding_data_handler.add_control_point(idx, x, y)

        self.adding_constraint = True
        self._parent_view.control_points = formatted
        self.adding_constraint = False

        self._parent_view.plot.update()

    def _add_must_link(self, event=None) -> None:
        """
        Callback for adding a new must-link.

        Args:
            event: Panel button click event.
        """

        formatted = self._embedding_data_handler.add_must_link(
            self._widgets["must_link_idx_1_str"].value,
            self._widgets["must_link_idx_2_str"].value
        )

        self.adding_constraint = True
        self._parent_view.must_links = formatted
        self.adding_constraint = False

        self._parent_view.plot.update()

    def _add_cannot_link(self, event=None) -> None:
        """
        Callback for adding a new cannot-link.

        Args:
            event: Panel button click event.
        """

        formatted = self._embedding_data_handler.add_cannot_link(
            self._widgets["cannot_link_idx_1_str"].value, 
            self._widgets["cannot_link_idx_2_str"].value
        )

        self.adding_constraint = True
        self._parent_view.cannot_links = formatted
        self.adding_constraint = False

        self._parent_view.plot.update()

    def _clear_control_points(self, event=None) -> None:
        self._parent_view.control_points = []

    def _clear_must_links(self, event=None) -> None:
        self._parent_view.must_links = []

    def _clear_cannot_links(self, event=None) -> None:
        self._parent_view.cannot_links = []

    def _deleted_links(self, event=None) -> list[tuple[str, str]]:
        """
        Determine which link pairs (must-links or cannot-links) were deleted
        between old and new display states.

        Args:
            event: Panel value-change event containing old and new values.

        Returns:
            list[tuple[str, str]]: Pairs of deleted links.
        """

        old_pairs = set(tuple(map(str.strip, s.split("⟷"))) for s in event.old)
        new_pairs = set(tuple(map(str.strip, s.split("⟷"))) for s in event.new)
        return list(old_pairs - new_pairs)

    def _clear_constraints(self, event=None) -> None:
        """
        Clear all constraints (control points, must-links, cannot-links).

        Args:
            event: Triggering event (unused).
        """

        self._clear_control_points()
        self._clear_must_links()
        self._clear_cannot_links()