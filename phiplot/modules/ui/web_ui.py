from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import holoviews as hv
import panel as pn
from phiplot.modules.ui import *

if TYPE_CHECKING:
    from phiplot.main import App

# Set global Holoviews and Panel extensions
hv.extension("bokeh")
pn.extension("floatpanel", "modal", notifications=True)

logger = logging.getLogger(__name__)
logging.getLogger("bokeh").setLevel(logging.ERROR)

class WebUI:
    """
    Main entry point for the application's interactive web interface.

    The `WebUI` class manages:
    - Theme selection and styling
    - Initialization and configuration of all UI panels
    - Layout composition (header, menu, left/center/right columns)
    - Application launch logic and developer mode
    - Synchronization between the UI and the application backend

    This class integrates all modular panels into a cohesive Panel
    `BootstrapTemplate` layout.

    Args:
        app (App): The application backend instance providing data and logic.
        use_developer_mode (bool, optional): If True, launch the app
            immediately in developer mode without theme selection.
            Defaults to False.
    """

    def __init__(self, app: App, use_developer_mode: bool = False, theme: str = "default"):
        self.app = app
        self.data_handler = self.app.data_handler
        self.embedding_handler = self.app.embedding_handler
        self.window_destination = pn.Column()
        self.styling = Styling()
        self.session_id = app.session_id
        self.use_developer_mode = use_developer_mode
        self.theme = theme
        pn.config.theme = self.theme

    @property
    def state(self) -> dict:
        return None
    
    @state.setter
    def state(self, new_state) -> None:
        pass

    @property
    def base_menu(self):
        return [self.menus["data"], self.menus["filters"]]
    
    @property
    def base_info(self):
        return [
            ("Database Status", self.menus["data"].display_panes["db_status"]),
            ("Collection Info", self.menus["data"].display_panes["collection_info"]),
            ("Applied Filters", self.menus["filters"].widgets["filter_display"]),
        ]
    
    def build(self):
        self.main_view_toggle = pn.widgets.RadioButtonGroup(
            options=["Data Summary", "Clustering", "Embedding"],
            button_style="outline",
            sizing_mode="stretch_both"
        )
        self.main_view_toggle.param.watch(self._on_main_view_selection, "value")

        self.select_view_row = pn.Row(
            self.main_view_toggle,
            sizing_mode="stretch_width",
            height=self.styling.top_menu_height,
            styles=dict(background=self.styling.neutral_gray),
        )

        self.modals = dict(
            restart = RestartModal(self.app),
            help = HelpModal(),
            about = AboutModal()
        )

        self.menus = dict(
            data = DataMenu(self),
            filters = FilterMenu(self)
        )

        if self.theme == "dark":
            self.bokeh_theme = self.styling.bokeh_dark
        else:
            self.bokeh_theme = self.styling.bokeh_light

        self._build_header()
        self._build_footer()

        self.views = dict(
            data_summary = DataSummaryView(self),
            clustering = ClusteringView(self),
            embedding = EmbeddingView(self)
        )

        self.contents = pn.Column()
        self.contents.objects = self.views["data_summary"].view

        template = pn.template.BootstrapTemplate(
            title="",
            theme=self.theme,
            favicon="phiplot/assets/media/favicon.ico"
        )
        template.header.append(self.header)
        template.main.append(self.contents)

        self._overwrite_title(template, title="PhiPlot")
        self._warn_about_refersh(template)
        
        self.view = template

        if self.use_developer_mode:
            self.menus["data"]._on_connect_to_server()
            self.menus["data"]._on_connect_collection(db="geckoq_prototype", collection="molecules")
            self.menus["data"]._on_fetch_data(fetch_type="random_sample")
            self.menus["data"]._on_generate_fps()
            self.views["clustering"].cluster()
            pn.state.notifications.info("Developer mode is on.")

    def update_available_features(self):
        self.menus["filters"].update_available_features()
        self.views["embedding"].update_available_features()

    def _on_main_view_selection(self, event=None) -> None:
        value = self.main_view_toggle.value
        if "Data Summary" in value:
            self.contents.objects = self.views["data_summary"].view
        elif "Clustering" in value:
            self.contents.objects = self.views["clustering"].view
        elif "Embedding" in value:
            self.contents.objects = self.views["embedding"].view

    def _build_header(self) -> None:
        """
        Construct the application header row.
        """

        self._theme_toggle = pn.widgets.ToggleIcon(
            description="Theme",
            active_icon = "sun-filled",
            icon="moon-filled",
            value=True,
            size="2.5em"
        )
        self._theme_toggle.param.watch(lambda event: self._toggle_theme(), "value")

        self.header = pn.Row(
            pn.Column(
                pn.VSpacer(),
                pn.pane.PNG("phiplot/assets/media/logo.png", width=50),
                pn.VSpacer(),
            ),
            pn.pane.Markdown("# PhiPlot"),
            pn.HSpacer(),
            pn.Column(pn.VSpacer(), pn.pane.Markdown(f"session id: {self.session_id}"), pn.VSpacer()),
            pn.HSpacer(),
            pn.Row(
                #pn.Column(pn.VSpacer(), self._theme_toggle, pn.VSpacer()),
                pn.Column(pn.VSpacer(), self.modals["about"].button, pn.VSpacer()),
                pn.Column(pn.VSpacer(), self.modals["help"].button, pn.VSpacer()),
                pn.Column(pn.VSpacer(), self.modals["restart"].button, pn.VSpacer())
            )
        )

    def _build_footer(self) -> None:
        self.footer = pn.Column(
            pn.Row(
                pn.HSpacer(),
                pn.pane.Markdown(
                    "Developed within the [EDA group](https://www.helsinki.fi/en/researchgroups/exploratory-data-analysis) \
                    at the University of Helsinki. Part of the [CoE VILMA](https://www.helsinki.fi/en/researchgroups/vilma)."
                ),
                pn.HSpacer()
            )
        )
    
    def _overwrite_title(self, template: pn.template, title: str) -> None:
        """
        A workaround to force a custom title to be shown for the 
        browser tab when using a template to build the app.

        Args:
            template (pn.template): The template to modify.
            title (str): The title to use.

        Returns:
            pn.pane.HTML: An invisible HTML pane with injected JavaScript
        """

        js_code = f"""
        <script>
        window.document.title = "{title}";
        </script>
        """
        title_script = pn.pane.HTML(js_code, width=0, height=0)
        template.main.append(title_script)

    def _warn_about_refersh(self, template):
        js_code = """
        <script>
        window.addEventListener("beforeunload", function (e) {
            e.preventDefault();
            e.returnValue = "";
        });
        </script>
        """
        template.main.append(pn.pane.HTML(js_code, width=0, height=0))

    def _toggle_theme(self) -> None:
        return