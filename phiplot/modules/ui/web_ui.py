from __future__ import annotations
import asyncio
import random
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

# Load templates stylesheet
with open("phiplot/assets/templates/styles.css", "r") as f:
    custom_css = f.read()
pn.extension(raw_css=[custom_css])

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
        self.template = pn.template.BootstrapTemplate(
            title="",
            theme=self.theme,
            favicon="phiplot/assets/media/favicon.ico"
        )

        self._overwrite_title(self.template, title="PhiPlot")

        self.loading_messages = [
            "Reticulating splines...",
            "Constructing additional pylons...",
            "Swapping time and space...",
            "Granting wishes...",
            "Adjusting flux capacitor...",
            "Locating the required pixels...",
            "Brewing coffee...",
            "Counting backwards from Infinity...",
            "Spinning the hamster...",
            "Shovelling coal into the server...",
            "Initializing the initializer...",
            "Optimizing the optimizer..."
        ]

        splash_image = pn.pane.PNG(
            "phiplot/assets/media/teaser.png", 
            width=900,
            align="center"
        )

        self.text_styles = {
            'text-align': 'center', 
            'font-style': 'italic', 
            'font-size': '2em',
            'font-weight': 'bold',
            'color': "#7e8186",
            'transition': 'opacity 0.5s ease-in-out'
        }

        initial_msg = random.choice(self.loading_messages)
        self.loading_text_pane = pn.pane.HTML(
            initial_msg, 
            align="center",
            styles={**self.text_styles, 'opacity': '1'}
        )
        
        self.splash_layout = pn.Column(
            pn.VSpacer(),
            pn.Row(pn.HSpacer(), splash_image, pn.HSpacer()),
            self.loading_text_pane,
            pn.VSpacer(),
            sizing_mode='stretch_both',
            styles={'transition': 'opacity 1s ease-out', 'opacity': '1'}
        )

        self.header_container = pn.Row()
        self.main_container = pn.Column(self.splash_layout, sizing_mode='stretch_both')

        self.template.header.append(self.header_container)
        self.template.main.append(self.main_container)

        self.is_loading = True

        self.view = self.template
        pn.state.onload(self._transition_to_main_ui)

    async def _cycle_loading_messages(self):
        while self.is_loading:
            await asyncio.sleep(1)
            if not self.is_loading: 
                break
            
            self.loading_text_pane.styles = {**self.text_styles, 'opacity': '0'}
            await asyncio.sleep(0.5)
            if not self.is_loading: 
                break
        
            new_msg = random.choice(self.loading_messages)
            self.loading_text_pane.object = new_msg
            
            self.loading_text_pane.styles = {**self.text_styles, 'opacity': '1'}

    async def _transition_to_main_ui(self):
        asyncio.create_task(self._cycle_loading_messages())

        await asyncio.sleep(2)

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

        self.is_loading = False

        self.splash_layout.styles = {'transition': 'opacity 1s ease-out', 'opacity': '0'}
        await asyncio.sleep(1)

        self.contents = pn.Column(sizing_mode='stretch_both')
        self.contents.objects = self.views["data_summary"].view

        self.header_container.objects = [self.header]
        self.main_container.objects = [self.contents]

        self._warn_about_refresh(self.template)

        self.modals["about"].modal.toggle()

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
            pn.pane.HTML(
                '<h1 style="font-variant: small-caps; font-family: sans-serif; font-size:3em; font-weight: bold; margin: 0">PhiPlot</h1>',
            ),
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

    def _warn_about_refresh(self, template):
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