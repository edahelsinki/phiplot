from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .data_handler import DataHandler
    from phiplot.modules.embedding.embedding_handler import *

logger = logging.getLogger(__name__)


class EmbeddingDataHandler:
    def __init__(self, data_handler: DataHandler, embedding_handler: EmbeddingHandler):
        self._data_handler = data_handler
        self._embedding_handler = embedding_handler
        self._molecule_handler = data_handler._molecule_handler

        self._colorable_dtypes = ["int", "float", "bool", "categorical"]
        self._tooltip_dtypes = ["str", "int", "float", "bool", "categorical"]

        self._data: pd.DataFrame | None = None
        self._plot_df: pd.DataFrame | None = None

        self._fingerprint: str | None = None

        self._vdims: list[str] | None = None
        self._tooltip_columns: list[str] | None = None

    @property
    def algortim(self) -> str | None:
        return self._embedding_handler.algorithm

    @property
    def columns(self) -> list[str]:
        return self._data_handler.columns
    
    @property
    def colorby_columns(self) -> list[str]:
        column_dtypes = self._data_handler.column_dtypes
        return [
            col for col in self._data_handler.columns 
            if column_dtypes[col] in self._colorable_dtypes
        ]

    @property
    def filtered_indices(self) -> list[str] | None:
        return self._data_handler.filtered_indices
    
    @property
    def fp_dimensions(self) -> int:
        X = self._embedding_handler.X
        if X is not None:
            return len(X)

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint
    
    @fingerprint.setter
    def fingerprint(self, fp) -> None:
        if self._data_handler.data is None:
            logger.warning("Did not set the fingerprint for an empty dataset.")
            return

        supported = self._molecule_handler.supported_generators
        if fp in supported:
            self._fingerprint = fp
            X = np.vstack(self._data_handler.fingerprints[fp].values)
            self._embedding_handler.X = X
        else:
            logger.error(f"{fp} is not a supported fingerprint. Supported fingerprints: {supported}")

    @property
    def index_column(self):
        return self._data_handler.index_column

    @property
    def indices(self) -> list[str]:
        return self._data_handler.indices

    @property
    def plot_df(self) -> pd.DataFrame | None:
        if self._plot_df is not None:
            return self._plot_df

    @property
    def smiles_like_cols(self):
        return self._data_handler.smiles_like_cols

    @property
    def tooltip_columns(self) -> list[str]:
        if self._tooltip_columns:
            return self._tooltip_columns.copy()
        return []
    
    @tooltip_columns.setter
    def tooltip_columns(self, cols) -> None:
        if all([col in self._columns for col in cols]):
            self._tooltip_columns = cols
        else:
            logger.error(
                "Could not set the tooltip columns as at least one of the provided columns is not present in the data..."
            )
    
    @property
    def vdims(self) -> list[str] | None:
        if self._vdims is not None:
            return self._vdims.copy()
        return []

    def remove_filtered_points(self):
        self._data_handler.remove_filtered_points()

    def create_plot_df(self) -> None:
        """
        Construct the DataFrame used for plotting.

        This includes:
        - Generating necessary plotting metadata columns (e.g., point fill alpha, line colors).
        - Determining which columns should be included in hover tooltips.
        - Handling special cases like SMILES/image columns and index columns.
        """

        try:
            # Create DataFrame from embedding coordinates and join with existing data
            df = pd.DataFrame(self._embedding_handler.embedding)
            df = df.join(self._data_handler.data)
            n = len(df)

            # Ensure index-column is of string type
            df[self._data_handler.index_column] = df[self._data_handler.index_column].astype(str)

            # Add columns controlling fill-alpha and line-color for
            # datapoints and include molecule-image-column in vdims
            df["fill_alpha"] = [1.0] * n
            df["line_color"] = [None] * n
            self._vdims = self.tooltip_columns + ["fill_alpha", "line_color", "img"]
            self._plot_df = df

            self._tooltip_columns = []
            column_dtypes = self._data_handler.column_dtypes
            for col in self.columns:
                # Include only numeric or string columns for hover-tooltips
                if column_dtypes[col] in self._tooltip_dtypes:
                    self._vdims.append(col)
                    self._tooltip_columns.append(col)

        except Exception as e:
            logging.exception("Error durring plotting DataFrame creation:")

    def pos_to_custom_idx(self, idx: int) -> str | None:
        """
        Map a positional index (row number) to its custom index.

        Args:
            idx (int): Row position in DataFrame.

        Returns:
            (str | None): Corresponding custom index value, or None if
                non-integer idx is provided.
        """

        if not isinstance(idx, int):
            log_process(
                ProcessResult(False, "The idx must be an integer", True, "error")
            )
            return

        return self._data_handler.data.at[idx, self._data_handler.index_column]

    def custom_to_pos_idx(self, idx: str) -> int | None:
        """
        Map a custom index back to its positional row index.

        Args:
            idx (str): Custom index value.

        Returns:
            (int | None): Row index in the DataFrame, or None if non-string idx is provided
                or idx cannot be found in the index-column.
        """

        if not isinstance(idx, str):
            logger.error("The idx must be a string.")

        idx = idx.strip()
        if idx not in self._data_handler.data[self._data_handler.index_column].values:
            logger.error(f"{idx} not found in {self._data_handler.index_column}")

        mask = self._data_handler.data[self._data_handler.index_column] == idx
        return self._data_handler.data.index.get_loc(mask.idxmax())

    def format_control_points(self) -> list[str]:
        """
        Format control points into human-readable strings.

        Returns:
            list[str]: Control points as `index: (x, y)` strings.
        """

        formatted = []
        for point in list(self._embedding_handler.control_points.items()):
            idx = self.pos_to_custom_idx(point[0])
            if idx:
                formatted.append(
                    f"{idx}: ({float(point[1][0]):.2f}, {float(point[1][1]):.2f})"
                )
        return formatted

    def add_control_point(self, idx: str, x: float, y: float) -> list[str]:
        """
        Add a control point to the embedding.

        Args:
            idx (str): Custom index.
            x (float): X coordinate.
            y (float): Y coordinate.

        Returns:
            list[str]: Updated list of formatted control points.
        """

        try:
            pos_idx = self.custom_to_pos_idx(idx)
            self._embedding_handler.add_control_point(pos_idx, x, y)
        except KeyError:
            logger.error("Invalid index {idx}")
        return self.format_control_points()

    def remove_control_points(self, deleted: list) -> list[str]:
        """
        Remove one or more control points.

        Args:
            deleted (list): List of custom indices to remove.

        Returns:
            list[str]: Updated formatted control points.
        """

        deleted = [self.custom_to_pos_idx(idx) for idx in deleted]
        if deleted:
            self._embedding_handler.remove_control_points(deleted)
        return self.format_control_points()

    def format_links(self, links: list[set[int]]) -> list[str]:
        """
        Format index pairs as bidirectional link strings.

        Args:
            links (list[set[int]]): List of index pairs.

        Returns:
            list[str]: Formatted link strings.
        """

        formatted = []
        for link in links:
            i, j = link
            idx1 = self.pos_to_custom_idx(i)
            idx2 = self.pos_to_custom_idx(j)
            if idx1 and idx2:
                formatted.append(f"{str(min(idx1, idx2))} ⟷ {str(max(idx1, idx2))}")
        return formatted

    def add_must_link(
        self, idx1: str | int, idx2: str | int, positional: bool = False
    ) -> list[str]:
        """
        Add a must-link constraint between two indices.

        Args:
            idx1 (str|int): Index of the first point.
            idx2 (str|int): Index of the second point.
            positional (bool): If True, do not convert to custom index.

        Returns:
            list[str]: Updated formatted must-links.
        """

        return self._add_link(idx1, idx2, "must", positional)

    def remove_must_link(self, deleted: list) -> list[str]:
        """
        Remove the must-link constraints between the listed pairs of indices.

        Args:
            deleted (list[set(str, str)]): The list of index pairs.

        Returns:
            list[str]: Updated formatted must-links.
        """

        return self._remove_links(deleted, "must")

    def add_cannot_link(
        self, idx1: str | int, idx2: str | int, positional: bool = False
    ) -> list[str]:
        """
        Add a cannot-link constraint between two indices.

        Args:
            idx1 (str|int): Index of the first point.
            idx2 (str|int): Index of the second point.
            positional (bool): If True, do not convert to custom index.

        Returns:
            list[str]: Updated formatted cannot-links.
        """

        return self._add_link(idx1, idx2, "cannot", positional)

    def remove_cannot_link(self, deleted: list) -> list[str]:
        """
        Remove the cannot-link constraints between the listed pairs of indices.

        Args:
            deleted (list[set(str, str)]): The list of index pairs.

        Returns:
            list[str]: Updated formatted cannot-links.
        """

        return self._remove_links(deleted, "cannot")

    def search_by_index(self, search_index: str) -> int | None:
        """
        Search for a row position by custom index.

        Args:
            search_index (str): Index value to search for.

        Returns:
            (int | None): Position in DataFrame, or None if not found.
        """

        if self._data_handler.data is not None and isinstance(search_index, str):
            search_index = search_index.strip()
            if search_index not in self.indices:
                logger.error(f"Could not find the index {search_index} in the embedding.")  
                return
            return self.custom_to_pos_idx(search_index)

    def _normalize_link(self, idx1: str, idx2: str) -> tuple[int, int] | None:
        """
        Normalize a link by converting custom indices to positional indices.

        Args:
            idx1 (str): First custom index.
            idx2 (str): Second custom index.

        Returns:
            (tuple[int, int] | None): Normalized pair, or None if invalid.
        """

        if idx1 == idx2:
            logger.error(f"Cannot create/delete a link between the same index: {idx1}")
            return
        try:
            idx1 = self.custom_to_pos_idx(idx1)
            idx2 = self.custom_to_pos_idx(idx2)
        except KeyError:
            logger.error(f"Invalid index in ({idx1}, {idx2})")
            return
        return tuple(sorted((idx1, idx2)))

    def _add_link(
        self, idx1: str | int, idx2: str | int, link_type: str, positional: bool = False
    ) -> list[str]:
        """
        Add a link constraint between two indices.

        Args:
            idx1 (str|int): Index of the first point.
            idx2 (str|int): Index of the second point.
            link_type (str): Type of the link (must or cannot)
            positional (bool): If True, do not convert to custom index.

        Returns:
            list[str]: Updated formatted links or an empty list if
                the link type is invalid.
        """

        if not positional:
            pair = self._normalize_link(str(idx1), str(idx2))
        else:
            pair = tuple(sorted((int(idx1), int(idx2))))
        if pair:
            if link_type.lower() == "must":
                self._embedding_handler.add_must_link(pair)
                return self.format_links(self._embedding_handler.must_links)
            elif link_type.lower() == "cannot":
                self._embedding_handler.add_cannot_link(pair)
                return self.format_links(self._embedding_handler.cannot_links)
            else:
                logger.error(f"Unsupported link type: {link_type}. Should be either 'must' or 'cannot'.")
                return []

    def _remove_links(self, deleted: list[set[str, str]], link_type: str) -> list[str]:
        """
        Remove the link constraints between the listed pairs of indices.

        Args:
            deleted (list[set(str, str)]): The list of index pairs.
            link_type (str): Type of the link (must or cannot)

        Returns:
            list[str]: Updated formatted links or an empty list if
                the link type is invalid.
        """

        deleted = [self._normalize_link(pair[0], pair[1]) for pair in deleted]
        if deleted:
            if link_type.lower() == "must":
                self._embedding_handler.remove_must_links(deleted)
                return self.format_links(self._embedding_handler.must_links)
            elif link_type.lower() == "cannot":
                self._embedding_handler.remove_cannot_links(deleted)
                return self.format_links(self._embedding_handler.cannot_links)
            else:
                logger.error(f"Unsupported link type: {link_type}. Should be either 'must' or 'cannot'.")
                return []