from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING
import logging
import panel as pn
import param
from jinja2 import Environment, FileSystemLoader
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.ui.widgets import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI

logger = logging.getLogger(__name__)

# Load HTML templates
env = Environment(loader=FileSystemLoader("phiplot/assets/templates"))


class DataMenu(BaseMenu):
    """
    UI panel group for managing fetching, importing and exporting data.
    Integrates and synchronizes with the backend `DataHandler`

    Args:
        app (App): Reference to the main application instance.
        window_destination (pn.Column): Destination container for rendering floating panels.
    """

    def __init__(self, parent_view: WebUI):
        super().__init__(
            parent_view=parent_view,
            name="Data",
            icon="database"
        )

        self._floating_panels = dict(
            select_collection_panel = self._build_select_collection_panel(),
            collection_settings_panel = self._build_collection_settings_panel(),
            fetch_data_panel = self._build_fetch_data_panel(),
            load_dataset_panel = self._build_load_dataset_panel(),
            generate_fps_panel = self._build_generate_fps_panel(),
            generate_rdkit_panel = self._build_generate_rdkit_panel(),
            import_state_panel = self._build_import_state_panel(),
            export_state_panel = self._build_export_state_panel()
        )

        self._data_handler.connect_tqdm(self._tqdm_struct, self._tqdm_fp, self._tqdm_rdkit)

        self.display_panes = dict(
            collection_info = self._build_collection_info_section(),
            db_status = self._build_db_status_section()
        )

        self.menu_items = [
            ("Connect to Server", "connect_server"),
            ("Select Collection", "select_collection"),
            ("Collection Settings", "collection_settings"),
            None,
            ("Fetch Data", "fetch_data"),
            #("Load External Dataset", "load_dataset"),
            None,
            ("Generate Fingerprints", "generate_fps"),
            ("Generate RDKit Features", "generate_rdkit"),
            #None,
            #("Import State", "import_settings"),
            #("Export State", "export_settings")
        ]

        self.callbacks = dict(
            connect_server = lambda: self._on_connect_to_server(),
            select_collection = lambda: self._floating_panels["select_collection_panel"].open(),
            collection_settings = lambda: self._floating_panels["collection_settings_panel"].open(),
            fetch_data = lambda: self._floating_panels["fetch_data_panel"].open(),
            load_dataset = lambda: self._floating_panels["load_dataset_panel"].open(),
            generate_fps = lambda: self._floating_panels["generate_fps_panel"].open(),
            generate_rdkit = lambda: self._floating_panels["generate_rdkit_panel"].open(),
            import_settings = lambda: self._floating_panels["import_state_panel"].open(),
            export_embedding = lambda: self._floating_panels["export_embedding_panel"].open(),
            export_settings =  lambda: self._floating_panels["export_state_panel"].open(),
        )

    @contextmanager
    def toggle_db_indicator(self):
        """
        Context manager for toggling the database boolean status indicator,
        setting its color, and setting the status text.
        """

        self._db_status_indicator.value = False
        self._db_satus_text_pane.objects = [pn.Column(
            pn.HSpacer(),
            pn.pane.Markdown("Connecting..."),
            pn.HSpacer()
        )]
        
        try:
            yield
        finally:
            client_status = self._data_handler.get_client_status()

            if not client_status["has_credentials"]:
                color = "danger"
                text = "Invalid credentials..."
            elif client_status["is_connected"]:
                if not client_status["can_fetch"]:
                    color = "warning"
                    text = "Cannot fetch..."
                else:
                    color = "success"
                    text = "Connected!"
            else:
                color = "danger"
                text = "Lost connection..."

            self._db_status_indicator.color = color
            self._db_satus_text_pane.objects = [pn.Column(
                pn.HSpacer(),
                pn.pane.Markdown(text),
                pn.HSpacer()
            )]
            self._db_status_indicator.value = True

    def _on_connect_to_server(self):
        res = self._data_handler.connect_to_server()

        with self.toggle_db_indicator():
            if res.success:
                self._widgets["database_selector"].options = self._data_handler.available_databases
                first_db = self._data_handler.available_databases[0]
                firts_collection = self._data_handler.get_available_collections(
                    first_db
                )[0]
                self._widgets["database_selector"].value = first_db
                self._widgets["collection_selector"].value = firts_collection

    def _build_db_status_section(self) -> None:
        self._db_status_indicator = pn.indicators.BooleanStatus(
            align="center",
            color="light",
            width=25,
            height=25,
            value=False
        )

        self._db_satus_text_pane = pn.Column(
            pn.HSpacer(),
            pn.pane.Markdown("Waiting Connection..."),
            pn.HSpacer()
        )
        
        return pn.Row(
            self._db_status_indicator,
            self._db_satus_text_pane
        )

    def _build_select_collection_panel(self) -> WindowPanel:
        """
        Build the database collection selection panel.

        Includes:
        - Database selector
        - Collection selector for selected database
        - Collection connection button
        """

        self._widgets["database_selector"] = pn.widgets.Select(
            name="Database", options=[], sizing_mode="stretch_width"
        )
        self._widgets["database_selector"].param.watch(self._update_collection_options, "value")

        self._widgets["collection_selector"] = pn.widgets.Select(
            name="Collection", sizing_mode="stretch_width"
        )

        self._widgets["connect_collection_button"] = pn.widgets.Button(
            name="Connect to Collection",
            **self._styling.default_button_style
        )
        self._widgets["connect_collection_button"].on_click(self._on_connect_collection)

        window = WindowPanel("Collection", self._window_destination)
        window.contents = [
            self._widgets["database_selector"],
            self._widgets["collection_selector"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["connect_collection_button"],
        ]
        return window

    def _update_collection_options(self, event: param.parameterized.Event | None = None) -> None:
        self._widgets["collection_selector"].options = self._data_handler.get_available_collections(
            self._widgets["database_selector"].value
        )

    def _on_connect_collection(self, event=None, db: str | None = None, collection: str | None = None):
        if db is None:
            db = self._widgets["database_selector"].value
        if collection is None:
            collection = self._widgets["collection_selector"].value
        
        res = self._data_handler.set_collection(db, collection)
        self._notify_collection_settings_saved = False

        if res.success:
            self._update_collection_settings_opts()
            self._update_collection_info()
            self._parent_view.update_available_features()
            self._parent_view.views["data_summary"].update_options()
            self._notify_collection_settings_saved = True

    def _update_collection_settings_opts(self) -> None:
        self._widgets["index_field_selector"].options = self._data_handler.columns
        self._widgets["smiles_field_selector"].options = self._data_handler.columns
        self._widgets["fetch_fields_multichoice"].options = self._data_handler.columns
        self._guess_columns()

    def _guess_columns(self) -> None:
        self._widgets["index_field_selector"].value = self._data_handler.guess_column("molecule_index")
        self._widgets["smiles_field_selector"].value = self._data_handler.guess_column("SMILES")
        self._widgets["fetch_fields_multichoice"].value = self._data_handler.guess_relevant_fetch_cols()
        self._on_save_collection_settings()

    def _build_collection_settings_panel(self) -> WindowPanel:
        """
        Build the database collection settings panel.

        Includes:
        - Index field selector
        - SMILES field selector
        - Fetch fields multiselector
        - Save settings button
        """

        self._widgets["index_field_selector"] = pn.widgets.Select(
            name="Index Field", options=[], sizing_mode="stretch_width"
        )

        self._widgets["smiles_field_selector"] = pn.widgets.Select(
            name="Smiles Field", options=[], sizing_mode="stretch_width"
        )

        self._widgets["fetch_fields_multichoice"] = pn.widgets.MultiChoice(
            name="Fields to Fetch", options=[], sizing_mode="stretch_width"
        )

        self._widgets["save_collection_settings_button"] = pn.widgets.Button(
            name="Save Settings", **self._styling.default_button_style
        )
        self._widgets["save_collection_settings_button"].on_click(self._on_save_collection_settings)

        window = WindowPanel("Collection Settings", self._window_destination, width=520)
        window.contents = [
            self._widgets["index_field_selector"],
            self._widgets["smiles_field_selector"],
            self._widgets["fetch_fields_multichoice"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["save_collection_settings_button"],
        ]
        return window
    
    def _on_save_collection_settings(self, event=None) -> None:
        if self._data_handler.collection_set:
            idx_col = self._widgets["index_field_selector"].value
            index_res = self._data_handler.set_index_column(idx_col)
            smiles_res = self._data_handler.set_smiles_column(
                self._widgets["smiles_field_selector"].value
            )
            fetch_res = self._data_handler.set_fetch_cols(
                self._widgets["fetch_fields_multichoice"].value
            )
            results = [index_res, smiles_res, fetch_res]

            if all([res.success for res in results]):
                self._update_collection_info()
                self.initial_set = True
                if self._notify_collection_settings_saved:
                    pn.state.notifications.success("Collection settings saved succesfully!")
            else:
                if self._notify_collection_settings_saved:
                    pn.state.notifications.error("Error in saving collection settings...")
        else:
            pn.state.notifications.warning("No collection has been selected.")

    def _build_fetch_data_panel(self) -> WindowPanel:
        """
        Build the molecule fetching panel.

        Includes:
        - Fetching strategy selector
        - Possible fetching stratgey options
        - Fetch button and related loading indicator
        """

        self._widgets["fetching_method_selector"] = pn.widgets.Select(
            name="Select Molecules by",
            options=[
                "Random Sample",
                "By Filters",
                "Index Range",
                "Index Set from a File",
                "All",
            ],
            value="Random Sample",
            sizing_mode="stretch_width",
        )

        self._widgets["number_of_samples_int"] = pn.widgets.IntInput(
            name=f"Number of Samples (Maximum is {self._data_handler.client_doc_cap})",
            value=1000,
            step=10,
            start=1,
            end=self._data_handler.client_doc_cap,
            sizing_mode="stretch_width",
        )

        self._widgets["index_start_int"] = pn.widgets.IntInput(
            name="Start", value=0, step=100, start=0, sizing_mode="stretch_width"
        )

        self._widgets["index_end_int"] = pn.widgets.IntInput(
            name="End", value=1000, step=100, start=0, sizing_mode="stretch_width"
        )

        self._widgets["index_file"] = pn.widgets.FileInput(
            accept=".txt", sizing_mode="stretch_width"
        )

        self._fetch_opts = pn.bind(
            self._update_fetch_opts,
            self._widgets["fetching_method_selector"]
        )

        self._widgets["fetching_data_spinner"] = pn.indicators.LoadingSpinner(
            value=False,
            **self._styling.default_spinner_style
        )

        self._widgets["fetch_data_button"] = pn.widgets.Button(
            name="Fetch Data", **self._styling.default_button_style
        )
        self._widgets["fetch_data_button"].on_click(self._on_fetch_data)

        self._fetch_progress = ProgressBar(
            desc="Fetching from database", unit="molecule", width=480
        )
        self._data_handler.connect_fetch_progress(self._fetch_progress)

        window = WindowPanel("Fetch Data", self._window_destination, width=520)
        window.contents = [
            self._widgets["fetching_method_selector"],
            self._fetch_opts,
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["fetch_data_button"],
                self._widgets["fetching_data_spinner"]
            ),
            pn.Spacer(height=self._styling.default_spacer_height),
            self._fetch_progress.build(),
        ]
        return window
    
    def _update_fetch_opts(self, event=None) -> pn.Row:
        """
        Return widgets matching the current fetch type.
        """

        sel = self._widgets["fetching_method_selector"].value
        ws = []
        if sel == "Random Sample":
            ws = [self._widgets["number_of_samples_int"]]
        elif sel == "Index Range":
            ws = [
                self._widgets["index_start_int"],
                self._widgets["index_end_int"]
            ]
        elif sel == "Index Set from a File":
            ws = [self._widgets["index_file"]]

        return pn.Row(*ws)
    
    def _on_fetch_data(self, event=None, fetch_type: str | None = None) -> None:
        """
        Fetch data from the backend according to the current selection.
        Updates UI status indicators based on success or failure.

        Args:
            event: Panel button click event.
        """

        with toggle_spinner(self._widgets["fetching_data_spinner"]):
            with self.toggle_db_indicator():
                self._fetch_progress.empty()

                if fetch_type is None:
                    fetch_type = self._widgets["fetching_method_selector"].value.lower().replace(" ", "_")

                if fetch_type == "random_sample":
                    kwargs = dict(size=self._widgets["number_of_samples_int"].value)
                elif fetch_type == "index_set_from_a_file":
                    file_name = self._widgets["index_file"].name
                    file_bytes = self._widgets["index_file"].value
                    fetch_type = "index_set"
                    kwargs = dict(
                        index_set=self._data_handler.read_index_set(file_name, file_bytes)
                    )
                elif fetch_type == "index_range":
                    kwargs = dict(
                        self._widgets["index_start_int"],
                        self._widgets["index_end_int"]
                    )
                else:
                    kwargs = dict()

                fetch_res = self._data_handler.fetch(fetch_type, **kwargs)

    def _build_load_dataset_panel(self) -> WindowPanel:
        """
        Build the dataset loading panel.

        Includes:
        - File upload (CSV).
        - Load button.
        """

        self._widgets["external_data_file"] = pn.widgets.FileInput(
            accept=".csv", multiple=False, sizing_mode="stretch_width"
        )

        self._widgets["load_external_data_button"] = pn.widgets.Button(
            name="Load", **self._styling.default_button_style
        )
        self._widgets["load_external_data_button"].on_click(self._on_load_dataset)

        window = WindowPanel("Load Data", self._window_destination, width=520)
        window.contents = [
            self._widgets["external_data_file"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["load_external_data_button"],
        ]
        return window
    
    def _on_load_dataset(self, event=None) -> None:
        """
        Handle dataset upload and pass it to the data handler.
        Updates UI status indicators based on success or failure.

        Args:
            event: Panel button click event.
        """

        result = self._data_handler.read_bytes(
            self._widgets["external_data_file"].name, 
            self._widgets["external_data_file"].value
        )

        if result.success:
            self._update_collection_settings_opts()

        self._parent_view.views["embedding"].recompute_kernel_heuristics()
        self._parent_view.views["embedding"].flush_kernel_mpds()

    def _set_externel_mol_data(self, event=None) -> None:
        res = self._data_handler.set_external_mol_data(self._get_fp_gen_params())

    def _get_fp_gen_params(self) -> None:
        """
        Fetch the fingerprint generation parameter values from the widgets.
        """

        result = {}
        for gen, params in self.fp_gen_param_widgets.items():
            result[gen] = {p: w.value for p, w in params.items()}
        return result

    def _build_generate_fps_panel(self) -> WindowPanel:
        """
        Build the fingerprint generation panel.

        Includes:
        - Fingerprinting options
        - Progress indicators for:
            - fetching
            - 2D structure generation
            - fingerprint generation
        """

        self._fp_params = self._create_fp_param_section()

        self._widgets["generating_fps_spinner"] = pn.indicators.LoadingSpinner(
            value=False, **self._styling.default_spinner_style
        )

        self._widgets["generate_fps_button"] = pn.widgets.Button(
            name="Generate Fingerprints", **self._styling.default_button_style
        )
        self._widgets["generate_fps_button"].on_click(self._on_generate_fps)

        self._tqdm_struct = pn.widgets.Tqdm(write_to_console=True, sizing_mode="stretch_width")
        self._tqdm_fp = pn.widgets.Tqdm(write_to_console=True, sizing_mode="stretch_width")

        self._tqdm_struct.text = "Waiting for 2D structure generation to start..."
        self._tqdm_fp.text = "Waiting for fingerprint generation to start..."

        window = WindowPanel("Generate Fingerprints", self._window_destination, width=520)
        window.contents = [
            pn.Accordion(("Fingerprint Parameters", self._fp_params)),
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["generate_fps_button"],
                self._widgets["generating_fps_spinner"]
            ),
            pn.Spacer(height=self._styling.default_spacer_height),
            self._tqdm_struct,
            self._tqdm_fp
        ]
        return window
    
    def _create_fp_param_section(self) -> pn.Accordion:
        """
        Dynamically create `pn.Column` containing inputs for setting
        the parameters controlling fingerprint generation based on the
        available generators in the `MoleculeHandler` class.
        """

        fp_gen_params_section = pn.Accordion()
        fp_gen_params_widgets = {}
        gen_info = self._data_handler.get_fp_gen_info()

        for gen in gen_info["available_generators"]:
            int_opts = pn.Column()
            bool_opts = pn.Column()
            fp_gen_params_widgets[gen] = {}
            for p, val in gen_info["generator_defaults"][gen].items():
                if type(val) == int:
                    w = self._create_int_input(p, val)
                    int_opts.append(w)
                elif type(val) == bool:
                    w = self._create_bool_input(p, val)
                    bool_opts.append(w)
                else:
                    continue
                fp_gen_params_widgets[gen][p] = w

            int_sect = pn.Column(pn.VSpacer(), int_opts, pn.VSpacer())
            bool_sect = pn.Column(pn.VSpacer(), bool_opts, pn.VSpacer())
            fp_gen_params_section.append((gen, pn.Row(int_sect, bool_sect)))

        self.fp_gen_param_widgets = fp_gen_params_widgets

        return fp_gen_params_section

    def _create_int_input(self, name, default) -> pn.widgets.IntInput:
        return pn.widgets.IntInput(
            name=name, value=default, step=1, start=0, sizing_mode="stretch_width"
        )

    def _create_bool_input(self, name, default) -> pn.widgets.Checkbox:
        return pn.widgets.Checkbox(
            name=name, value=default, sizing_mode="stretch_width"
        )

    def _on_generate_fps(self, event=None) -> None:
        with toggle_spinner(self.widgets["generating_fps_spinner"]):
            self._tqdm_struct.value = 0
            self._tqdm_struct.text = "Waiting for 2D structure generation to start..."

            self._tqdm_fp.value = 0
            self._tqdm_fp.text = "Waiting for fingerprint generation to start..."

            self._data_handler.add_fingerprints(self._get_fp_gen_params())
            self._parent_view.views["embedding"].recompute_kernel_heuristics()
            self._parent_view.views["embedding"].flush_kernel_mpds()

    def _build_generate_rdkit_panel(self) -> WindowPanel:
        self._widgets["smarts_feature_name_str"] = pn.widgets.TextInput(
            name="Feature Name", sizing_mode="stretch_width"
        )

        self._widgets["smarts_pattern_str"] = pn.widgets.TextInput(
            name="SMARTS String", sizing_mode="stretch_width"
        )

        self._widgets["smarts_pattern_file"] = pn.widgets.FileInput(
            accept=".json", multiple=False, sizing_mode="stretch_width"
        )

        self._widgets["generating_rdkit_spinner"] = pn.indicators.LoadingSpinner(
            value=False, **self._styling.default_spinner_style
        )

        self._widgets["generate_rdkit_button"] = pn.widgets.Button(
            name="Generate RDKit Features", **self._styling.default_button_style
        )
        self._widgets["generate_rdkit_button"].on_click(self._on_generate_rdkit_features)

        self._tqdm_rdkit = pn.widgets.Tqdm(write_to_console=True, sizing_mode="stretch_width")
        self._tqdm_rdkit.text = "Waiting for RDKit feature generation to start..."

        window = WindowPanel("Generate RDKit Features", self._window_destination, width=520)
        window.contents = [
            pn.Row(
                self._widgets["smarts_feature_name_str"],
                self._widgets["smarts_pattern_str"]
            ),
            #pn.pane.Markdown("SMARTS Input File", styles=dict(margin_bottom="-8pt")),
            #self._widgets["smarts_pattern_file"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["generate_rdkit_button"],
                self._widgets["generating_rdkit_spinner"]
            ),
            pn.Spacer(height=self._styling.default_spacer_height),
            self._tqdm_rdkit
        ]
        return window

    def _on_generate_rdkit_features(self, event=None):
        name = self._widgets["smarts_feature_name_str"].value
        pattern = self._widgets["smarts_pattern_str"].value

        if len(name) == 0:
            name = pattern
            
        SMARTS = {name: pattern}

        with toggle_spinner(self.widgets["generating_fps_spinner"]):
            self._tqdm_rdkit.value = 0
            self._tqdm_struct.text = "Waiting for feature generation to start..."

            self._data_handler.generate_rdkit_features(SMARTS)
            self._parent_view.update_available_features()

    def _build_import_state_panel(self) -> WindowPanel:
        """
        Build the state import panel.

        Includes:
        - State file input field
        - State import button and related loading indicator
        """

        self._widgets["state_input_file"] = pn.widgets.FileInput(
            accept=".json", multiple=False, sizing_mode="stretch_width"
        )

        self._widgets["importing_state_spinner"] = pn.indicators.LoadingSpinner(
            value=False, **self._styling.default_spinner_style
        )

        self._widgets["import_state_button"] = pn.widgets.Button(
            name="Import State", **self._styling.default_button_style
        )
        self._widgets["import_state_button"].on_click(self._on_import_state)

        window = WindowPanel("Import State", self._window_destination)
        window.contents = [
            self._widgets["state_input_file"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["import_state_button"],
                self._widgets["importing_state_spinner"] 
            )
        ]
        return window
    
    def _on_import_state(self, event=None) -> None:
        """
        Import state and build the plot.

        Args:
            event: Panel button click event.
        """
        return
        with toggle_spinner(self._widgets["importing_state_spinner"]):
            file_bytes = self._widgets["state_input_file"]
            if file_bytes is None:
                pn.state.notifications.warning("No import file loaded.")
                return

            res = self._data_handler.exporter_importer.import_settings(
                self._widgets["state_input_file"].name, file_bytes
            )

            with toggle_indicator(
                self.app.ui.database_status, self._db_indicator_color
            ):
                plot_res = self.app.interactive_plot.build()
                emb_panels = self.app.ui.menus["embedding"]
                emb_panels.set_algorithm(self._embedding_handler.algorithm)

    def _build_export_state_panel(self) -> WindowPanel:
        """
        Build the state export panel.

        Includes:
        - State export button
        """

        self._widgets["state_output_file"] = pn.widgets.FileDownload(
            label="Export State",
            button_type="success",
            filename="settings.json",
            sizing_mode="stretch_width",
            callback=lambda: None
        )

        window = WindowPanel("Export State", self._window_destination)
        window.contents = [self._widgets["state_output_file"]]
        return window

    def _build_collection_info_section(self):
        self._collection_info_display = pn.pane.HTML(sizing_mode="stretch_width")
        self._update_collection_info()
        return pn.Column(self._collection_info_display)
    
    def _update_collection_info(self):
        template = env.get_template("simple_table.html")
        info = self._data_handler.get_collection_info()
        info = {field.replace("_", " ").title(): value for field, value in info.items()}
        html = template.render(info=info)
        self._collection_info_display.object = html
