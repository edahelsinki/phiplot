from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .data_handler import DataHandler

logger = logging.getLogger(__name__)

class ClusteringDataHandler:
    def __init__(self, data_handler: DataHandler):
        self._data_handler = data_handler
        self._cluster_label_column = "cluster_label"

    @property
    def cluster_label_column(self):
        return self._cluster_label_column

    @property
    def clustering_data(self) -> pd.DataFrame | None:
        if self._data_handler.data is not None:
            if self._cluster_label_column in self._columns:
                return self._data.drop(self._cluster_label_column, axis=1)
            else:
                return self._data.copy()
        return None