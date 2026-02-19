from __future__ import annotations
from typing import TYPE_CHECKING
from difflib import SequenceMatcher
import logging
from io import StringIO
from typing import Any
import numpy as np
import pandas as pd
from pandas.api import types
import panel as pn
from rdkit import Chem, RDLogger
from .molecule_handler import MoleculeHandler
from .embedding_data_handler import EmbeddingDataHandler
from .clustering_data_handler import ClusteringDataHandler
from phiplot.modules.data.db.client import DBClient
from phiplot.modules.ui.widgets import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.embedding.embedding_handler import EmbeddingHandler

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Manages data retrieval, preprocessing, filtering, and integration with embedding visualizations.

    Acts as the central interface for managing data flow between the database,
    embedding handler, and visualization pipeline.

    Args:
        use_local_db (bool): Whether to use a local database connection.
        embedding_handler (EmbeddingHandler): Embedding handler for managing
            feature matrices and dimensionality reduction.
    """

    def __init__(self, use_local_db: bool, embedding_handler: EmbeddingHandler) -> None:
        self._embedding_handler = embedding_handler
        self._molecule_handler = MoleculeHandler()

        self.embedding_data_handler = EmbeddingDataHandler(self, embedding_handler)
        self.clustering_data_handler = ClusteringDataHandler(self)

        self._client_doc_cap = 100000
        self._client = DBClient(
            use_local=use_local_db, document_cap=self._client_doc_cap
        )

        self._current_database = None
        self._current_collection = None

        self._sample = None
        self._data = None

        self._columns = None
        self._column_dtypes = None
        self._index_column = None
        self._smiles_column = None
        self._smiles_like_cols = None
        self._fetch_columns = None
        self._cluster_label_column = "cluster_label"

        self._filter_by_columns = []
        self._generated_cols = []

        self._added_indices = set()
        self._filtered_indices = set()
        self._filters = dict()

        self.supported_filter_types = [
            "range",
            "equal_to_number",
            "equal_to_categorical",
            "less_than",
            "greater_than",
        ]

        self.use_external_data = False
        self.use_external_smiles = False
        self.collection_set = False
        self.data_initialized = False
        self.plot_data_initialized = False
        self.deleting_filter = False

        self._default_index_col = "index"
        self._default_smiles_col_like = "SMILES"

        self._fingerprints: pd.DataFrame | None = None
        self._n_datapoints: int = 0

    @property
    def fingerprints(self) -> pd.DataFrame | None:
        return self._fingerprints

    @property
    def available_databases(self) -> list[str]:
        return list(self._client.available_collections.keys())

    @property
    def client_doc_cap(self) -> int:
        return self._client_doc_cap

    @property
    def cluster_label_column(self):
        return self._cluster_label_column

    @property
    def columns(self) -> list[str]:
        if self._columns:
            return list(set(self._columns.copy()))
        return []
    
    @property
    def column_dtypes(self) -> dict[str, str]:
        if self._column_dtypes:
            return self._column_dtypes.copy()
        return {}

    @property
    def data(self) -> pd.DataFrame | None:
        if self._data is not None:
            return self._data
        return None
    
    @property
    def clustering_data(self) -> pd.DataFrame | None:
        if self._data is not None:
            if self._cluster_label_column in self._columns:
                return self._data.drop(self._cluster_label_column, axis=1)
            else:
                return self._data.copy()
        return None

    @property
    def fetch_columns(self) -> list[str]:
        if self._fetch_columns:
            return list(set(self._fetch_columns.copy()))
        return []

    @property
    def db_data_filters(self) -> dict[str, dict]:
        return self._db_data_filters.copy()

    @property
    def filters(self) -> dict[str, dict]:
        return self._filters.copy()

    @property
    def filter_by_columns(self) -> list:
        return list(set(self._filter_by_columns.copy()))

    @property
    def filtered_indices(self) -> list[str] | None:
        return list(self._filtered_indices.copy())

    @property
    def index_column(self) -> str | None:
        return self._index_column

    @property
    def indices(self) -> list:
        if self._data is not None:
            return list(self._data[self._index_column])
        return []
    
    @property
    def n_data_points(self):
        return self._n_datapoints

    @property
    def smiles_like_cols(self) -> list[str]:
        if self._smiles_like_cols:
            return self._smiles_like_cols.copy()
        return []
    
    def add_features(self, new_features: pd.DataFrame, index_col_name: str) -> None:
        if isinstance(new_features, pd.DataFrame):
            try:
                merged = pd.merge(
                    self._data,
                    new_features,
                    left_on=self._index_column,
                    right_on=index_col_name,
                    how="inner"
                )
                
                if index_col_name != self._index_column:
                    merged.drop(index_col_name, axis=1, inplace=True)
                
                new_cols = [col for col in new_features.columns if col != index_col_name]
                for col in new_cols:
                    old_col = f"{col}_x"
                    if old_col in merged.columns:
                        merged.drop(old_col, axis=1, inplace=True)
                
                self._data = merged
                self._columns = list(set(self._columns + new_cols))
                for feature in new_features.columns:
                    self._column_dtypes[feature] = self._categorize_dtype(new_features[feature])
                self._filter_by_columns = list(set(self._filter_by_columns + new_cols))
                self._generated_cols = list(set(self._generated_cols + new_cols))
            except Exception:
                logger.exception(f"Could not add the new features:")
        else:
            logger.error("The new features should be given as a dataframe...")

    def _categorize_dtype(self, series: pd.Series, cat_threshold: float=0.05) -> str:
        """
        Categorizes Series into int, float, bool, str, categorical, or object.
        Includes logic to 'detect' categoricals based on unique value density.
        """

        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"
        
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        
        if pd.api.types.is_integer_dtype(series):
            if (series.nunique() / len(series)) < cat_threshold:
                return "categorical"
            return "int"
            
        if pd.api.types.is_float_dtype(series):
            return "float"
        
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 1
            
            if unique_ratio < cat_threshold:
                return "categorical"
            
            if pd.api.types.is_string_dtype(series):
                return "str"
            return "object"

        return "other"

    def generate_rdkit_features(self, SMARTS:dict[str, str]) -> ProcessResult:
        if self.data is None:
            res = ProcessResult(
                False, "Fetch molecules before generating features", True, "warning"
            )
        else:
            try:
                df = self._molecule_handler.generate_rdkit_features(
                    self._data, SMARTS
                )
                cols = list(SMARTS.keys())
                self._columns.extend(cols)
                self._filter_by_columns.extend(cols)
                self._generated_cols.extend(cols)
                self._data = df
                res = ProcessResult(
                    True, "Feature generation finished successfully!", True, "info"
                )
            except Exception as e:
                logger.exception("Error during generating RDKit features:")
                res = ProcessResult(
                    False, "An error occured during generating RDKit features...", True, "error"
                )
        log_process(res)
        return res

    def connect_tqdm(
        self, structures: pn.widgets.Tqdm, fingerprints: pn.widgets.Tqdm, features: pn.widgets.Tqdm
    ) -> None:
        """
        Connect the tqdm widgets to the molecule handler.

        Args:
            structures (pn.widgets.Tqdm): The tqdm widget for the strucutre generation.
            fingerprints (pn.widgets.Tqdm): The tqdm widget for the fingerprpint generation.
        """

        self._molecule_handler.set_tqdm(structures, fingerprints, features)

    def connect_fetch_progress(self, progress_bar: ProgressBar) -> None:
        """
        Connect the progress bar widget to the client.

        Args:
            progress_bar(ProgressBar): The progress bar widget for the fetching.
        """

        self._client.progress_bar = progress_bar

    def get_fp_gen_info(self) -> dict[str, Any]:
        """
        Get the available fingerprint generators and their default parameters.

        Returns:
            (dict[str, Any]): Constructed as:
                - "available_generators" (list[str]): Names of the supported generators.
                - "generator_defaults" (dict): Default parameters for each generator.
        """

        return dict(
            available_generators=self._molecule_handler.supported_generators,
            generator_defaults=self._molecule_handler.generator_params,
        )

    def get_client_status(self) -> dict[str, bool]:
        """
        Get the status of the client.

        Returns:
            (dict[str, bool]): Constructed as:
                - "has_credentials" (bool): True if the database credentials are valid.
                - "is_connected" (bool): True if the database can be pinged.
                - "can_fetch" (bool): True if the last fetching operation was succesfull.
        """

        return dict(
            has_credentials=self._client.has_credentials,
            is_connected=self._client.is_connected,
            can_fetch=self._client.can_fetch,
        )

    def get_available_collections(self, database: str) -> list[str]:
        """
        List all the available collections within a database.

        Args:
            database (str): The name of a database on the server.

        Returns:
            list[str]: The names of the available collections within the database.
        """

        if not database:
            return []
        return self._client.available_collections[database]

    def set_collection(self, database: str, collection: str) -> ProcessResult:
        """
        Choose the collection of a database within the server.

        Args:
            database (str): The name of a database on the server.
            collection (str): The name of a collection within the database.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self._client.set_collection(database, collection)

        collection_fields = self._client.fields
        self._columns = list(collection_fields.keys())
        self._filter_by_columns = self._columns
        self._generated_cols = []
        self._column_dtypes = collection_fields

        self._current_database = database
        self._current_collection = collection

        if self._client.is_connected:
            res = ProcessResult(
                True, "Collection connected successfully!", True, "info"
            )
            self.collection_set = True
        else:
            res = ProcessResult(
                False, "Could not connect to the collection.", True, "error"
            )
        log_process(res)

        return res

    def get_collection_info(self) -> dict[str, str | None]:
        """
        Get info about the currently selected collection. This inlcudes:
        - Name of the database
        - Name of the collection
        - Name of the selected index column
        - Name of the selected SMILES column, if applicable
        """

        return dict(
            database=self._current_database,
            collection=self._current_collection,
            index_column=self._index_column,
            smiles_column=self._smiles_column,
        )

    def get_non_index_columns(self) -> list[str]:
        """
        Get all but the index column for the current collection.

        Returns:
            list: All data columns except the index column if it has been set.
        """

        cols = []
        if self._data is not None:
            cols = list(self._data.columns)
        if self._index_column is not None:
            cols.remove(self._index_column)
        return cols

    def get_numerical_summary(self, search_field: str, use_filters: bool = False, n_buckets: int = 10) -> dict[str, Any]:
        filters = None
        if use_filters:
            filters = self.filters
        summary = self._client.numerical_field_summary(search_field, filters, n_buckets)

        return dict(
            edges = summary["distribution"]["edges"],
            counts = summary["distribution"]["counts"],
            summary_stats = summary["summary_stats"]
        )
    
    def get_categorical_summary(self, search_field: str, use_filters: bool = False) -> dict[str, Any]:
        filters = None
        if use_filters:
            filters = self.filters
        summary = self._client.categorical_field_summary(search_field, filters)

        return dict(
            labels = summary["distribution"]["labels"],
            counts = summary["distribution"]["counts"],
            summary_stats = summary["summary_stats"]
        )
    
    def get_categorical_comparison(
            self,
            cat_field: str, 
            labels: list[str], 
            comparison_field: str, 
            use_filters: bool = False
        ) -> dict[str, dict]:
        
        base_filters = {}
        if use_filters:
            base_filters = self.filters
        
        dtype = self.column_dtypes[cat_field] 
        if dtype == "categorical_int":
            labels = list(map(int, labels))
        
        summaries = {}
        for label in labels:
            addtional_filter = {"feature": cat_field, "dtype": dtype, "type": "equal_to", "value": label}
            filters = base_filters | {self._format_filter(addtional_filter): addtional_filter}
            query = self._client.filters_to_query(filters)
            summaries[label] = self._client._get_numerical_field_summary_stats(comparison_field, query)
        return summaries
        
    def set_index_column(self, column: str) -> ProcessResult:
        """
        Set the column to use as the index.

        Args:
            column (str): The column name.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        if self._validate_column(column):
            self._index_column = column
            self._client.index_field = column
            if self._data is not None:
                self._data[column].astype("str")
            res = ProcessResult(True, "Index column set.", False, "debug")
        else:
            res = ProcessResult(
                False, "Could not set the index column.", False, "debug"
            )
        log_process(res)
        return res

    def set_smiles_column(self, column: str) -> ProcessResult:
        """
        Set the column to use for fingerprint generation.

        Args:
            column (str): The column name.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        if self._validate_column(column):
            self._smiles_column = column
            self._client.smiles_field = column
            res = ProcessResult(True, "SMILES column set.", False, "debug")
        else:
            res = ProcessResult(
                False, "Could not set the SMILES column.", False, "debug"
            )
        log_process(res)
        return res

    def set_fetch_cols(self, columns: list[str]) -> ProcessResult:
        """
        Set the columns to fetch for the projection operation.

        Args:
            columns (list[str]): The names of the columns.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        all_valid = all([self._validate_column(col) for col in columns])
        if all_valid:
            self._fetch_columns = columns
            self._client.set_projection(include=columns)
            res = ProcessResult(True, "Fetch columns set.", False, "debug")
        else:
            res = ProcessResult(
                False, "Could not set the fetch columns.", False, "debug"
            )
        log_process(res)
        return res

    def set_external_mol_data(self, generator_params: dict) -> ProcessResult:
        """
        Use a SMILES column in an external dataset to generate fingerprints for a sample.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self.use_external_smiles = True
        return self.add_fingerprints(generator_params)

    def connect_to_server(self) -> ProcessResult:
        """
        Connect the client to the server.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self._client.init_client()

        if not self._client.has_credentials:
            res = ProcessResult(
                False, "Improper credentials provided...", True, "error"
            )
        elif not self._client.is_connected:
            res = ProcessResult(False, "Could not connect the client...", True, "error")
        else:
            res = ProcessResult(True, "Client connected successfully!", True, "info")
        log_process(res)
        return res

    def fetch(self, fetch_type: str, **kwargs) -> ProcessResult:
        """
        Fetch data from the database according to a given strategy and store
        the results as a list inside `self._sample`.

        Args:
            fetch_type (str): Type of fetch operation. One of:
                - "random_sample": Random subset of docs.
                - "by_filters": Docs matching active filters.
                - "index_range": Docs with index in given range.
                - "index_set": Docs with index in given set.
                - "all": All docs in the database.
            **kwargs: Additional fetch parameters (e.g., size, start, end, file_name).

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        fetcher = {
            "random_sample": lambda: self._client.random_sample(
                kwargs.get("size", 1000)
            ),
            "by_filters": lambda: self._client.filtered_sample(self._filters),
            "index_range": lambda: self._client.index_range_sample(
                kwargs.get("start", 0), kwargs.get("end", 5000)
            ),
            "index_set": lambda: self._client.index_set_sample(
                kwargs.get("indices", set())
            ),
            "all": lambda: self._client.all_docs(),
        }

        self._client.ping()

        if not self._client.is_connected:
            res = ProcessResult(
                False, "Lost connection to the server...", True, "error"
            )
        else:
            self._sample = fetcher[fetch_type]()
            n = len(self._sample)
            if n == 0:
                if not self._client.can_fetch:
                    res = ProcessResult(False, "The collection is unfetchable...", True, "error")
                else:
                    res = ProcessResult(False, "No data was fetched.", True, "error")
                    self._n_datapoints = 0
            else:
                self.use_external_data = False
                self.use_external_smiles = False
                self.data_initialized = True

                if len(self._sample) >= self._client_doc_cap:
                    res = ProcessResult(
                        True,
                        f"Document limit reached. Showing first {self._client_doc_cap} documents.",
                        True,
                        "warning",
                    )
                    self._n_datapoints = self._client_doc_cap
                else:
                    res = ProcessResult(
                        True, "Documents fetched succesfully!", True, "info"
                    )
                    self._n_datapoints = n
        log_process(res)
        return res

    def add_fingerprints(self, generator_params: dict = {}) -> ProcessResult:
        """
        Add all the supported fingerprints for the fetched molecules and store
        the results as a `pd.DataFrame` inside `self._data`.

        Args:
            generator_params (dict): The paramaters controlling each fingerprint generator.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        if self._sample is None:
            res = ProcessResult(
                False, "Fetch molecules before adding fingerprints.", True, "warning"
            )
        else:
            try:
                if generator_params:
                    self._molecule_handler.generator_params = generator_params

                df = self._molecule_handler.sample_to_df(self._sample, self._smiles_column)

                df[self._index_column] = df[self._index_column].astype(str)

                self._smiles_like_cols = [col for col in self._columns if self._is_smiles_col(df[col])]

                gens = self._molecule_handler.supported_generators
                self._fingerprints = df[gens]
                df = df.drop(columns=gens)

                if "_id" in self._columns:
                    df = df.drop(columns=["_id"])

                self._data = df

                cols = list(self._data.columns)
                for col in cols:
                    self._column_dtypes[col] = self._categorize_dtype(self._data[col])

                self._column_dtypes["img"] = "img_link"
                self._columns = cols
                
                res = ProcessResult(
                    True, "Fingerprinting finished successfully!", True, "info"
                )
            except Exception as e:
                logger.exception("Error during fingerprinting:")
                res = ProcessResult(
                    False, "An error occured during fingerprinting...", True, "error"
                )
        log_process(res)
        return res

    def read_bytes(self, filename: str, bytes_: bytes) -> ProcessResult:
        """
        Read data from an uploaded CSV file into a DataFrame.

        Args:
            filename (str): Name of the file for logging/debugging.
            bytes_ (bytes): Raw CSV file content.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        try:
            data_str = bytes_.decode("utf-8")
            df = pd.read_csv(StringIO(data_str))
            if df.empty:
                res = ProcessResult(
                    False,
                    f"No data was read from the file {filename}.",
                    True,
                    "warning",
                )
            else:
                self._master_data = df
                self._sample = df
                self._columns = list(df.columns)
                self.use_external_data = True
                self.data_initialized = True
                res = ProcessResult(
                    True, f"File {filename} was read successfully!", True, "info"
                )
        except (
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as e:
            logger.exception("Error during reading the data:")
            res = ProcessResult(
                False, "An error occured during reading the data.", True, "error"
            )
        log_process(res)
        return res

    def _str_to_num(self, val):
        if "." in val:
            try:
                return float(val)
            except:
                return np.nan
        else:
            try:
                return int(val)
            except:
                return np.nan

    def add_filter(
        self, feature: str, filter_type: str, filter_options: list
    ) -> list[str] | None:
        """
        Add a filter condition to the dataset.

        Args:
            feature (str): Column name to filter on.
            filter_type (str): Type of filter (e.g., "range", "equal_to").
            filter_options (list): Parameters for the filter.

        Returns:
            list[str]: Active filter keys.
        """

        filter_dtype = self._get_filter_dtype(filter_type)
        filter_type = self._normalize_filter_type(filter_type)
        new_filter = {"feature": feature, "dtype": filter_dtype, "type": filter_type}

        try:
            if filter_type in ["equal_to", "less_than", "greater_than"]:
                new_filter["value"] = filter_options[0]
            elif filter_type == "set_of":
                if filter_dtype == "object":
                    new_filter["value"] = set([self._str_to_num(val.strip()) for val in filter_options[0].split(",")])
                else:
                    new_filter["value"] = set([val.strip() for val in filter_options[0].split(",")])
            elif filter_type == "range":
                new_filter["start"] = filter_options[0]
                new_filter["end"] = filter_options[1]
        except IndexError:
            log_process(
                ProcessResult(False, f"Filter is missing an argument.", False, "error")
            )

        if not filter_dtype is None:
            formatted = self._format_filter(new_filter)
            self._filters[formatted] = new_filter

            if self.data_initialized and self._data is not None:
                self._apply_filter(formatted, new_filter)

        return [*self._filters]

    def _get_filter_dtype(self, ft: str) -> str | None:
        if ft in ["range", "equal_to_number", "less_than", "greater_than"]:
            return "numeric"
        elif ft in ["equal_to_categorical", "set_of_categoricals"]:
            return "object"
        else:
            log_process(
                ProcessResult(
                    False,
                    f"Invalid filter type {ft}. Supported types are {self.supported_filter_types}",
                    False,
                    "error",
                )
            )
            return None

    def _normalize_filter_type(self, ft: str) -> str:
        if "equal" in ft:
            return "equal_to"
        elif "set_of" in ft:
            return "set_of"
        return ft

    def remove_filters(self, deleted: list) -> list:
        """
        Remove filter conditions from the dataset.

        Args:
            deleted (list[str]): Identifiers of the filters to remove.

        Returns:
            list[str]: Active filter keys aaer deletion.
        """

        for f in deleted:
            if f in self._filters:
                del self._filters[f]

        if self.data_initialized and self._data is not None:
            self._filtered_indices = set()
            for key, f in self.filters.items():
                self._apply_filter(key, f)

        return [*self._filters]
    
    def remove_filtered_points(self) -> None:
        """
        Permanently remove filtered points from the dataset and remap constraints.
        """

        try:
            current_labels = self._data.index
            to_drop = current_labels.intersection(self.filtered_indices)

            if not to_drop.empty:
                remaining_indices = current_labels.difference(self.filtered_indices)
                pos_map = {old_idx: new_pos for new_pos, old_idx in enumerate(remaining_indices)}
                self._data = self._data.drop(index=to_drop).reset_index(drop=True)
                self._fingerprints = self._fingerprints.drop(index=to_drop).reset_index(drop=True)
                self._embedding_handler.remap_constraints(pos_map)
                self._filtered_indices = set()
        except Exception as e:
            logger.exception(f"Error during removing filtered points:")

    def search_by_index(self, search_index: str) -> int | None:
        """
        Search for a row position by custom index.

        Args:
            search_index (str): Index value to search for.

        Returns:
            (int | None): Position in DataFrame, or None if not found.
        """

        if self._data is not None and isinstance(search_index, str):
            search_index = search_index.strip()
            if search_index not in self._data[self._index_column].values:
                log_process(
                    ProcessResult(
                        False,
                        f"Could not find the index {search_index} in the embedding.",
                        True,
                        "error",
                    )
                )
                return
            return self.custom_to_pos_idx(search_index)

    def read_index_set(self, name: str, file_bytes: bytes) -> set:
        """
        Parse a UTF-8 text file containing document indices (one per line).

        Args:
            name (str): Name of the file (used in logging).
            file_bytes (bytes): Raw byte content of the uploaded file.

        Returns:
            set: Indices read from the file.
        """

        if not file_bytes:
            log_process(
                ProcessResult(
                    False,
                    f"Could not read the file '{name}': empty or unreadable.",
                    True,
                    "error",
                )
            )
            return set()

        try:
            lines = file_bytes.decode("utf-8").splitlines()
            index_set = {int(line) for line in map(str.strip, lines) if line.strip()}
        except UnicodeDecodeError as e:
            log_process(
                ProcessResult(False, f"File '{name}' is not valid UTF-8", True, "error")
            )
            return set()
        except ValueError as e:
            log_process(
                ProcessResult(
                    False, f"Invalid line in file '{name}': {e}", True, "error"
                )
            )
            return set()

        if not index_set:
            log_process(
                ProcessResult(
                    False, f"File '{name}' contains no valid indices.", True, "warning"
                )
            )
        else:
            log_process(
                ProcessResult(
                    False,
                    f"Parsed {len(index_set)} indices from '{name}'.",
                    True,
                    "info",
                )
            )

        return index_set

    def guess_column(self, like: str) -> str:
        """
        Guess the most similar column name to the input string.

        Args:
            like (str): The target string to match.

        Returns:
            str: Most similar column name
        """

        if not self._columns:
            return None

        def normalize(s):
            return s.lower().replace("_", "").replace(" ", "")

        like_norm = normalize(like)

        sims = [
            (col, SequenceMatcher(None, normalize(col), like_norm).ratio())
            for col in self._columns
        ]

        col, score = max(sims, key=lambda x: x[1])
        return col

    def guess_relevant_fetch_cols(self) -> list[str]:
        """
        Guess which columns are relevant fetch columns by filtering
        out columns with dtypes `dict` and `list`.

        Returns:
            list: Names of the guessed relevant columns.
        """
        return [
            col
            for col in self._columns
            if (col[0] != "_" and self._column_dtypes[col] not in ["dict", "list"])
        ]
    
    def fetch_single_doc(self, idx: str) -> dict | None:
        try:
            doc = self._client.fetch_single_doc(int(idx))[0]
            doc.pop("_id", None)
            img_path = self._molecule_handler.mol_to_img(doc, self._smiles_column)
            return doc, img_path
        except:
            return None, None

    def _validate_column(self, column: str) -> bool:
        """
        Check that a column exists in the data.

        Args:
            column (str): The name of the column to check.

        Returns:
            (bool): True if the column exists in data,
                or None if it does not exist.
        """
        if column in self._columns:
            return True
        else:
            log_process(
                ProcessResult(
                    False, f"No column called {column} in data.", False, "warning"
                )
            )
            return False

    def _is_smiles_col(self, col_series: pd.Series, N_samples: int = 20) -> bool:
        """
        Infer whether a column contains valid SMILES-strings based on a random sample of rows.

        Args:
            col_series (pd.Series): The values within the column to check
            N_samples (int): The number of random samples to take.

        Returns:
            bool: True if all the tested values were valid
                SMILES-strings, False otherwise.
        """

        values = col_series.dropna().astype(str)
        lg = RDLogger.logger()
        if values.empty:
            return False
        sample = values.sample(min(N_samples, len(values)), random_state=42)
        for v in sample:
            lg.setLevel(RDLogger.CRITICAL)
            mol = Chem.MolFromSmiles(v)
            lg.setLevel(RDLogger.WARNING)
            if mol is None:
                return False
        return True

    def _format_filter(self, filter_spec: dict) -> str | None:
        """
        Convert a filter specification into a human-readable string.

        Args:
            filter_spec (dict): Filter specification.

        Returns:
            (str | None): String representation of filter,
                or None if the filter is invalid.
        """

        map_to_string = {
            "range": lambda: f"{filter_spec['start']} ≤ {filter_spec['feature']} ≤ {filter_spec['end']}",
            "equal_to": lambda: f"{filter_spec['feature']} = {filter_spec['value']}",
            "less_than": lambda: f"{filter_spec['feature']} < {filter_spec['value']}",
            "greater_than": lambda: f"{filter_spec['feature']} > {filter_spec['value']}",
            "set_of": lambda: f"{filter_spec['feature']} ϵ {filter_spec['value']}",
        }

        try:
            return map_to_string[filter_spec["type"]]()
        except:
            log_process(ProcessResult(False, f"Invalid filter type.", False, "error"))
            return

    def _apply_filter(self, key: str, filter_spec: dict) -> list[str]:
        """
        Apply a filter to the dataset.

        Args:
            key (str): Identifier for the filter.
            filter_spec (dict): Filter specification.

        Returns:
            list[str]: Active filter keys.
        """

        col = self._data[filter_spec["feature"]]
        ftype = filter_spec["type"]
        dtype = filter_spec["dtype"]
        comp = filter_spec["value"]

        if dtype == "object":
            col_processed = col.astype(str).str.lower().str.strip()
        else:
            col_processed = col

        map_to_operation = {
            "range": lambda: (col_processed >= filter_spec["start"]) & (col_processed <= filter_spec["end"]),
            "equal_to": lambda: col_processed == comp,
            "less_than": lambda: col_processed < comp,
            "greater_than": lambda: col_processed > comp,
            "set_of": lambda: col_processed.isin(comp),
        }

        keep_mask = map_to_operation[ftype]()
        hide_mask = ~keep_mask

        new_hidden_indices = set(self._data.index[hide_mask])
        self._filtered_indices = self._filtered_indices.union(new_hidden_indices)

        if hide_mask.size == 0:
            log_process(
                ProcessResult(False, "No matches, filter not applied", True, "warning")
            )
            del self._filters[key]
            return [*self._filters]
