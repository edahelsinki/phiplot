from collections import defaultdict
import logging
import os
from math import nan
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from phiplot.modules.ui.widgets import *

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DBClient:
    """
    A client for interacting with a MongoDB-backed database.

    This class provides functionality to connect to local or remote MongoDB instances,
    retrieve and query molecule records, and handle error reporting and connection validation.

    Args:
        use_local (bool): If True use a local databse instance, otherwise tries to connect
            to the remote database.
        local_uri (str): The uri for the local database instance.
        document_cap (int): The maximum number of documents the client will fetch.
    """

    def __init__(
        self,
        use_local: bool = False,
        local_uri: str = "mongodb://mongodb:27017",
        document_cap: int = 5000,
    ):
        self._use_local_db = use_local
        self._local_uri = local_uri
        self._document_cap = document_cap

        self.has_credentials = True
        self.is_connected = True
        self.can_fetch = True

        self._sample = None

        self._database = None
        self._collection = None
        self._available_collections = {}
        self._projection = None

        self._index_field = None
        self._smiles_field = None

        self._progress_bar = None

        self._cat_cutoff = 10

    @property
    def available_collections(self) -> dict[str, list[str]]:
        return self._available_collections

    @property
    def collection(self) -> str:
        return f"{self._collection.database.name}.{self._collection.name}"

    @property
    def document_cap(self) -> int:
        return self._document_cap

    @property
    def fields(self) -> dict[str, str]:
        if not hasattr(self, "_fields_cache"):
            self._fields_cache = self.find_fields()
        return self._fields_cache
    
    @property
    def simple_fields(self) -> list[str]:
        return [
            field for field, dtype in self.fields.items() 
            if dtype in ["int", "float", "str"]
        ]

    @property
    def complex_fields(self) -> list[str]:
        return [
            field for field, dtype in self.fields.items() 
            if dtype not in ["int", "float", "str"]
        ]

    @property
    def index_field(self) -> str:
        return self._index_field

    @index_field.setter
    def index_field(self, field: str) -> None:
        self._index_field = self.validate_field(field)

    @property
    def smiles_field(self) -> str:
        return self._smiles_field

    @smiles_field.setter
    def smiles_field(self, field: str) -> None:
        self._smiles_field = self.validate_field(field)

    @property
    def progress_bar(self) -> ProgressBar:
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar: ProgressBar) -> None:
        self._progress_bar = progress_bar

    @property
    def projection(self) -> dict[str, list[str]]:
        return {
            "included": [f for f, v in self._projection if v == 1],
            "excluded": [f for f, v in self._projection if v == 0],
        }

    def set_collection(self, db: str, collection: str) -> None:
        """
        Select a collection within a database from which to fetch the data.

        Args:
            db (str): The name of the database
            collection (str): A collection within the database
        """

        if db not in self._available_collections:
            logger.error(f"There is no database called {db}.")
            return

        if collection not in self._available_collections[db]:
            logger.error(f"There is no collection {collection} within {db}.")
            return

        self.ping()
        if self.is_connected:
            self._database = self.client[db]
            self._collection = self._database[collection]
            if hasattr(self, "_fields_cache"):
                del self._fields_cache

    def set_projection(self, include: list | None = None, exclude: list | None = None):
        """
        Set the projection to use when fetching data.

        Args:
            include (list): The fields to include in the projection.
            exclude (list): The fields to explicitly exclude in the projection.
        """

        include = include or []
        exclude = exclude or []

        invalid = [f for f in include + exclude if f not in self.fields.keys()]
        if invalid:
            logger.error(
                f"Invalid fields {invalid} in collection {self._collection.name}"
            )
            return

        self._projection = {
            **{f: 1 for f in include},
            **{f: 0 for f in exclude},
        }

    def init_client(self) -> None:
        """
        Initialize the MongoDB client by retrieving credentials from environment
        variables and establishing a connection to either a local or a remote database.
        """

        if self._use_local_db:
            self.has_credentials = True
            uri = self._local_uri
        else:
            self.has_all_env_vars = True
            uri_raw = self._get_first_env_var(
                ["MONGO_URI", "MONGODB_URI", "MONGOURI"], required=True
            )

            if self.has_all_env_vars:
                uri = f"{uri_raw}/?readPreference=secondaryPreferred"
            else:
                self.has_credentials = False
                return

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.ping()
        except Exception:
            return

        if self.has_credentials and self.is_connected:
            system_dbs = ["admin", "config", "local"]

            available_databases = self.client.list_database_names()
            for db in available_databases:
                if db not in system_dbs:
                    self._available_collections[db] = self.client[db].list_collection_names()

    def ping(self) -> None:
        """
        Checks the connection to the MongoDB server by issuing a ping command.

        If the connection fails, logs an error and sets `self.is_connected` to False.
        """

        try:
            self.client.admin.command("ping")
            logger.debug("Database connection is healthy.")
            self.is_connected = True
            self.has_credentials = True
        except errors.OperationFailure as e:
            logger.debug(f"Operation failure occurred during pinging: {e}")
            self.has_credentials = False
        except errors.PyMongoError as e:
            logger.debug(f"Generic error occurred during pinging: {e}")
            self.is_connected = False

    def find_fields(self, N_dtype: int = 100, N_cat: int = 500):
        """
        Infer all the available fields and their dtypes within a collection 
        by randomly sampling the documents. As additional dtypes, we consider 
        "categorical_str" and "categorical_int", which are string or integer 
        fields with the lower limit of the number of estimated unique values 
        less than `self._cat_cutoff`.

        Args:
            N_dtype (int): The number of random samples to infer dtype.
            N_cat: The number of random samples to infer categorical fields.

        Returns:
            (dict[str, str]): The inferred fields as keys and the most general
                inferred dtype for each field as the value.
        """
        if self._collection is None:
            return {}

        field_types = defaultdict(set)
        sampled_docs = list(self._collection.find().limit(N_dtype))
        field_sets = []

        for doc in sampled_docs:
            for field, value in doc.items():
                field_types[field].add(type(value))
                field_sets.append(set(doc.keys()))

        if not field_types:
            return {}
        elif field_sets and not all(f == field_sets[0] for f in field_sets):
            logger.warning(
                f"Not all documents share the same fields in {self.collection}."
            )

        inferred = {}
        for field, types in field_types.items():
            dtype = self._promote_types(field, types)
            field_type = dtype
            if dtype in ["str", "int"]:
                count = list(self._collection.aggregate([
                    {"$sample": {"size": N_cat}},
                    {"$project": {field: 1, "_id": 0}},
                    {"$group": { "_id": f"${field}"} },
                    {"$count": "uniqueCount" }
                ]))[0]["uniqueCount"]
                if count < self._cat_cutoff:
                    field_type = f"categorical_{dtype}"
            inferred[field] = field_type

        return inferred

    def validate_field(self, field: str) -> str | None:
        """
        Validates that a field exists in the current collection.

        Args:
            field (str): Name of the field to check for existence.

        Returns:
            (str | None): The field name if it exists in the collection,
                otherwise None.
        """

        available_fields = set(self.fields.keys())

        if not available_fields:
            logger.error(
                "There are no fields available. Remember to set the collection first."
            )
            return None
        elif field not in available_fields:
            logger.error(f"There is no field called {field} in the current collection.")
            return None

        return field
    
    def categorical_field_summary(self, search_field: str, filters: dict = None) -> dict[str, dict]:
        """
        Get the distribution of values and summary statistics of a categorical field.

        Args:
            search_field (str): The name of the field to analyse.
            filters (dict): Optional user-defined filters to filter documents by.
                Defaults to None.

        Returns:
            (dict[str, dict]): Constructed as:
                "distribution" (dict): The labels and counts for the bar plot
                "summary" (dict): The computed summary statistics.
                
                An empty dict is returned if the analysis fails.
        """

        if search_field not in self.fields:
            logger.error("The search field could not be found in the collection")
            return {}
        
        mongo_query = None
        if filters:
            mongo_query = self.filters_to_query(filters)

        dtype = self.fields[search_field]

        if "categorical" in dtype:
            distribution = self._get_categorical_field_distribution(search_field, mongo_query)
            summary_stats = self._get_categorical_field_summary_stats(search_field, mongo_query)
            if distribution and summary_stats:
                return dict(
                    distribution = distribution,
                    summary_stats = summary_stats
                )
        else:
            logger.error(f"Invalid field dtype {dtype} for categorical field summary.")
        return {}
        
    def numerical_field_summary(self, search_field: str, filters: dict = None, n_buckets: int = 10) -> dict[str, dict]:
        """
        Get the distribution of values and summary statistics of a numerical field.

        Args:
            search_field (str): The name of the field to analyse.
            filters (dict): Optional user-defined filters to filter documents by.
                Defaults to None.
            n_buckets (int): The number of buckets to use for the histrogram.
                Defaults to 10.

        Returns:
            (dict[str, dict]): Constructed as:
                "distribution" (dict): The labels and counts for the histogram plot
                "summary" (dict): The computed summary statistics.
                
                An empty dict is returned if the analysis fails.
        """

        if search_field not in self.fields:
            logger.error("The search field could not be found in the collection")
            return {}
        
        mongo_query = None
        if filters:
            mongo_query = self.filters_to_query(filters)

        dtype = self.fields[search_field]

        if dtype in ["float", "int"]:
            distribution = self._get_numerical_field_distribution(search_field, mongo_query, n_buckets)
            summary_stats = self._get_numerical_field_summary_stats(search_field, mongo_query)
            if distribution and summary_stats:
                return dict(
                    distribution = distribution,
                    summary_stats = summary_stats
                )
        else:
            logger.error(f"Invalid field dtype {dtype} for numerical field summary.")
        return {}
           
    def fetch(self, query: dict) -> list:
        """
        Query the MongoDB collection for documents matching the given criteria.

        Args:
            query (dict): A MongoDB query to filter documents.

        Returns:
            list: A list of fetched documents. If an error occurs, an empty list
                is returned and `self.data_fetch_error` is set to True.
        """

        try:
            cursor = self._collection.find(
                filter=query, projection=self._projection, limit=self._document_cap
            )

            result = []
            self._progress_bar.start()
            for doc in cursor:
                result.append(doc)
                self._progress_bar.update()
            self._progress_bar.finished()

            n_docs = len(result)

            if n_docs == self._document_cap:
                logger.warning(
                    (
                        "There might be more documents matching the query. "
                        f"Returning the first {self._document_cap} documents."
                    )
                )
            elif n_docs == 0:
                logger.warning("No documents matching the search.")

            self.can_fetch = True
            return result

        except errors.PyMongoError as e:
            logger.debug(f"Database error occured when fetching the documents: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error occured when fetching the documents: {e}")

        self._progress_bar.finished()
        self.can_fetch = False
        return []

    def random_sample(
        self, N: int, scale: int = 1.2, batch_fraction: float = 0.03
    ) -> list:
        """
        Fetch a random sample of N documents in batches.

        Args:
            N (int): The number of random molecules to retrieve.
            scale (int): The number of molecules to sample beyond N to account
                for null flexibility.
            batch_fraction (float): The proportion of documents in each batch.

        Returns:
            list: A list of randomly sampled molecule documents,
                or an empty list on error.
        """

        try:
            total_count = self._collection.estimated_document_count()
            batch_size = int(total_count * batch_fraction)
            seen_ids = set()
            sampled_docs = []

            if N < batch_size:
                batch_size = N
            elif N > self._document_cap:
                N = self._document_cap
                logger.warning(
                    (
                        "Sample is larger than the supported max document count. "
                        f"Sample size was reduced to {self._document_cap} documents."
                    )
                )

            self._progress_bar.start()
            while len(sampled_docs) < N:
                batch = list(
                    self._collection.aggregate(
                        [
                            {"$sample": {"size": int(batch_size * scale)}},
                            {"$project": self._projection},
                            {"$limit": batch_size},
                        ]
                    )
                )

                for doc in batch:
                    id = str(doc[self._index_field])
                    if id not in seen_ids:
                        seen_ids.add(id)
                        sampled_docs.append(doc)
                        if len(sampled_docs) >= N:
                            break
            self._progress_bar.finished()

            return sampled_docs
        except errors.PyMongoError as e:
            logger.debug(f"Database error occured when fetching the documents: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error occured when fetching the documents: {e}")

        self._progress_bar.finished()
        self.can_fetch = False
        return []

    def all_docs(self) -> list:
        """
        Fetch all documents.

        Returns:
            list: Fetched documents.
        """

        return self.fetch({})

    def filtered_sample(self, filters: dict[str, dict[str, str | int | float]]) -> list:
        """
        Build a MongoDB query from a set of user-defined filters and
        fetch all corresponding docs.

        Args:
            filters (dict[str, dict[str, str | int | float]]): A dictionary where each value contains:
                - feature (str): The field to query.
                - type (str): The type of filter (equal_to, less_than, etc.).
                - value/start/end: Required based on filter type.

        Returns:
            list: The fetched documents or an empty list if the query is not valid.
        """

        mongo_query = self.filters_to_query(filters)
        
        if mongo_query is None:
            logger.warning("No valid filters were parsed. Fetching all molecules.")
            return self.fetch({})

        return self.fetch(mongo_query)

    def index_set_sample(self, index_set: set) -> list:
        """
        Fetch the docs with index in the given set.

        Args:
            index_set (set): Indices of the docs to be fetched.

        Returns:
            list: Fetched documents.
        """

        if not index_set:
            logger.warning(f"No valid indices.")
            return []

        return self.fetch({self._index_field: {"$in": list(index_set)}})

    def index_range_sample(self, start: int, end: int) -> list:
        """
        Fetch the docs with indices between `start` and `end` (inclusive).

        Args:
            start (int): The starting index.
            end (int): The ending index.

        Returns:
            list: Fetched documents.
        """

        if start > end:
            logger.error(f"Invalid range: start ({start}) > end ({end})")
            return []

        return self.fetch({self._index_field: {"$gte": start, "$lte": end}})

    def fetch_single_doc(self, index: int) -> list:
        """
        Fetch one document by its index.

        Args:
            index (int): The index of the document to look up.

        Returns:
            list: A list containing the matching document,
                or an empty list if the query fails.
        """

        try:
            result = self._collection.find_one(
                {self._index_field: {"$eq": index}}, projection = self._projection
            )
            if result is None:
                logger.warning(f"Could not find a document with index {index}.")
                return []
            else:
                logger.info(f"Document with index {index} was fetched successfully!")

            return [result]

        except errors.PyMongoError as e:
            logger.error(
                f"Database error occured when fetching the document with index {index}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error occured when fetching the document with index {index}: {e}"
            )
        return []
    
    def filters_to_query(self, filters: dict[str, dict[str, str | int | float]]) -> dict[str, dict[str, str | int | float]] | None:
        """
        Build a MongoDB query from a set of user-defined filters.

        Args:
            filters (dict): A dictionary where each value contains:
                - feature (str): The field to query.
                - type (str): The type of filter (equal_to, less_than, etc.).
                - value/start/end: Required based on filter type.

        Returns:
            dict: The corresponding MongoDB query.
        """

        if filters is None:
            return None
        
        if len(list(filters.keys())) == 0:
            return None

        query = defaultdict(dict)

        for key, f in filters.items():
            feature = f.get("feature")
            filter_type = f.get("type")

            if not feature or not filter_type:
                logger.error(
                    f"Invalid filter format in '{key}': missing 'feature' or 'type'"
                )
                continue

            try:
                if filter_type == "equal_to":
                    query[feature]["$eq"] = f["value"]
                elif filter_type == "less_than":
                    query[feature]["$lt"] = f["value"]
                elif filter_type == "greater_than":
                    query[feature]["$gt"] = f["value"]
                elif filter_type == "range":
                    query[feature]["$gte"] = f["start"]
                    query[feature]["$lte"] = f["end"]
                elif filter_type == "set_of":
                    vals = f["value"]
                    query[feature]["$in"] = list(vals) if isinstance(vals, (set, list)) else [vals]
                else:
                    logger.error(f"Unknown filter type in '{key}': {filter_type}")
            except KeyError as e:
                logger.error(f"Missing key in filter '{key}': {e}")
                continue
        
        return {feature: condition for feature, condition in query.items()}

    def _get_first_env_var(
        self, possible_keys: list[str], default=None, required=False
    ) -> str | None:
        """
        Return the first non-empty environment variable value found in the given list of keys.

        Args:
            possible_keys (list[str]): A list of environment variable names to check, in order.
            default (str, optional): Value to return if none of the keys are found. Defaults to None.
            required (bool): If True and no key is found, raises an error. Defaults to False.

        Returns:
            (str | None): The first matching environment variable's value, or default if none are found.
        """
        for key in possible_keys:
            value = os.environ.get(key)
            if value:
                return value

        if required:
            self.has_all_env_vars = False
            logger.error(
                f"Missing required environment variable. Tried: {possible_keys}"
            )

        return default

    def _promote_types(self, field: str, types: list):
        """
        Resolve a set of Python types into the most general common dtype.

        Args:
            field (str): Name of the document field.
            types (list): The set of inferred dtypes for the field.

        Returns:
            str: The promoted dtype.
        """

        non_null = {t for t in types if t is not type(None)}
        if not non_null:
            return "None"

        if len(non_null) == 1:
            return next(iter(non_null)).__name__

        if non_null <= {int, float}:
            logger.debug(
                f"Field '{field}' had mixed types {non_null}, promoted to float."
            )
            return "float"

        logger.debug(f"Field '{field}' had mixed types {non_null}, promoted to str.")
        return "str"

    def _get_numerical_field_summary_stats(
            self, 
            search_field: str, 
            mongo_query: dict | None = None
        ) -> dict[str, int | float]:
        """
        Get the summary statistics of a numerical field.

        Args:
            search_field (str): The name of the field to get the stats from.
            mongo_query (str): An optional MongoDB query to filter docs by. Defaults to None.

        Returns:
            (dict[str, int | float]): Where the keys are the names of the
                statistics and values their computed values, or an empty dict 
                if the aggregation fails.
        """

        field_filter = {search_field: {"$type": "number", "$ne": nan}}
        mongo_query = mongo_query or {}

        pipeline = [
            {"$match": self._combine_filters(mongo_query, field_filter)},
            {
                "$facet": {
                    "summary": [
                        {
                            "$group": {
                                "_id": None,
                                "count": {"$sum": 1},
                                "min": {"$min": f"${search_field}"},
                                "max": {"$max": f"${search_field}"},
                                "avg": {"$avg": f"${search_field}"},
                                "stdDev": {"$stdDevPop": f"${search_field}"}
                            }
                        }
                    ],
                    "buckets": [
                        {
                            "$bucketAuto": {
                                "groupBy": f"${search_field}",
                                "buckets": 4,
                                "output": {"count": {"$sum": 1}}
                            }
                        }
                    ]
                }
            }
        ]

        result = list(self._collection.aggregate(pipeline))

        if not result:
            return {}

        summary = result[0]["summary"][0] if result[0]["summary"] else {}
        buckets = result[0]["buckets"]

        percentiles = {}
        percentile_keys = ["25%", "50%", "75%"]

        for i, key in enumerate(percentile_keys, start=1):
            if i < len(buckets):
                percentiles[key] = buckets[i]["_id"]["min"]
            else:
                percentiles[key] = None

        return {
            "count": summary.get("count", 0),
            "mean": summary.get("avg"),
            "std": summary.get("stdDev"),
            "min": summary.get("min"),
            **percentiles,
            "max": summary.get("max"),
            "dtype": self.fields[search_field]
        }

    
    def _get_categorical_field_summary_stats(
            self, 
            search_field: str, 
            mongo_query: dict | None = None
        ) -> dict[str, int | str]:
        """
        Get the summary statistics of a categorical field.

        Args:
            search_field (str): The name of the field to get the stats from.
            mongo_query (str): An optional MongoDB query to filter docs by. Defaults to None.

        Returns:
            (dict[str, int | str]): Where the keys are the names of the
                statistics and values their computed values, or an empty dict if
                the aggregation fails.
        """

        mongo_query = mongo_query or {}

        pipeline = [
            {"$match": {**mongo_query}},
            {
                "$facet": {
                    "summary": [
                        {
                            "$group": {
                                "_id": None,
                                "count": {"$sum": 1},
                                "unique": {"$addToSet": f"${search_field}"}
                            }
                        }
                    ],
                    "top_category": [
                        {"$group": {"_id": f"${search_field}", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": 1}
                    ]
                }
            }
        ]

        result = list(self._collection.aggregate(pipeline))

        if not result:
            return {}

        facet_data = result[0]

        summary = facet_data.get('summary', [{}])[0]
        top_category = facet_data.get('top_category', [{}])[0]

        return {
            "count": summary.get("count", 0),
            "unique": len(summary.get("unique", [])),
            "top": top_category.get("_id"),
            "freq": top_category.get("count"),
            "dtype": self.fields[search_field],
        }

    def _get_numerical_field_distribution(
            self,
            search_field: str,
            mongo_query: dict | None = None,
            n_buckets: int = 10
        ) -> tuple[list[str], list[int]]:
        """
        Get the distribution of values of a numerical field.

        Args:
            search_field (str): The name of the field to get the distribution from.
            mongo_query (str): An optional MongoDB query to filter docs by. Defaults to None.
            n_buckets (int): The number of buckets to use for the histogram. Defaults to 10.

        Returns:
            dict: 
        """

        field_filter = {search_field: {"$type": "number", "$ne": nan}}
        mongo_query = mongo_query or {}

        min_max_pipeline = [
            {"$match": self._combine_filters(mongo_query, field_filter)},
            {
                "$group": {
                    "_id": None,
                    "minValue": {"$min": f"${search_field}"},
                    "maxValue": {"$max": f"${search_field}"}
                }
            }
        ]

        min_max_result = list(self._collection.aggregate(min_max_pipeline))

        if not min_max_result:
            return [], []

        min_search_val = min_max_result[0]["minValue"]
        max_search_val = min_max_result[0]["maxValue"]

        step = (max_search_val - min_search_val) / n_buckets
        buckets = [min_search_val + i * step for i in range(n_buckets)] + [max_search_val + 1e-10]

        pipeline = [
            {"$match": self._combine_filters(mongo_query, field_filter)},
            {
                "$bucket": {
                    "groupBy": f"${search_field}",
                    "boundaries": buckets,
                    "default": "Other",
                    "output": {
                        "count": {"$sum": 1}
                    }
                }
            }
        ]

        result = list(self._collection.aggregate(pipeline))

        if not result:
            return {}
        
        full_edges = [(min_search_val + i * step, min_search_val + (i + 1) * step) for i in range(n_buckets)]
        result_dict = {doc["_id"]: int(doc["count"]) for doc in result if isinstance(doc["_id"], (int, float))}
        
        edges = []
        counts = []

        for start, end in full_edges:
            edges.append((start, end))
            counts.append(int(result_dict.get(start, 0)))

        return dict(
            edges = edges,
            counts = counts
        )
    
    def _get_categorical_field_distribution(
            self,
            search_field: str,
            mongo_query: dict | None = None
        ) -> tuple[list[str], list[int]]:
        """
        Get the distribution of values of a categorical field.

        Args:
            search_field (str): The name of the field to get the distribution from.
            mongo_query (str): An optional MongoDB query to filter docs by. Defaults to None.

        Returns:
            dict: 
        """

        mongo_query = mongo_query or {}

        pipeline = [
            {"$match": {**mongo_query}},
            {"$group": {"_id": f"${search_field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        result = list(self._collection.aggregate(pipeline))

        if not result:
            return {}

        labels = []
        counts = []

        for doc in result:
            bucket_id = doc["_id"]
            labels.append(str(bucket_id))
            counts.append(doc["count"])

        return dict(
            labels = labels,
            counts = counts,
        )
    
    def _combine_filters(self, mongo_query: dict, field_filter: dict) -> dict:
        if not mongo_query:
            return field_filter
        
        overlapping_keys = set(mongo_query.keys()) & set(field_filter.keys())
        
        if not overlapping_keys:
            return {**mongo_query, **field_filter}
        
        combined_conditions = []
        combined_conditions.append(mongo_query)
        combined_conditions.append(field_filter)
        return {"$and": combined_conditions}