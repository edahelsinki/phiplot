from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator

from phiplot.modules.ATMOMACCS import *

logger = logging.getLogger(__name__)


class MoleculeHandler:
    """
    Handles SMILES-based molecule processing including:
        - Generating 2D structure images
        - Generating molecular fingerprints (The list of all supported
          generators can be accessed via the `supported_generators` property)
        - Storing results into structured DataFrames
    """

    _supported_generators = {
        "Morgan": rdFingerprintGenerator.GetMorganGenerator,
        "RDKit": rdFingerprintGenerator.GetRDKitFPGenerator,
        "AtomPairs": rdFingerprintGenerator.GetAtomPairGenerator,
        "TopologicalTorsions": rdFingerprintGenerator.GetTopologicalTorsionGenerator,
        "ATMOMACCS": ATMOMACCSGenerator,
    }

    def __init__(self):
        self._generator_params = self._create_generator_defaults()
        self._tqdm_struct = None
        self._tqdm_fp = None
        self._tqdm_rdkit = None

    @property
    def supported_generators(self) -> list[str]:
        return list(self._supported_generators.keys())

    @property
    def generator_params(self) -> dict[str, dict]:
        return self._generator_params

    @generator_params.setter
    def generator_params(self, gen_params):
        """
        Set generator parameters for supported fingerprint types.

        Args:
            gen_params (dict): A mapping of fingerprint type → parameter dictionary.

        Raises:
            KeyError: If a fingerprint type is not in `self._supported_generators`
                or if a provided parameter is not valid for that generator.
        """
        for fp_type, fp_params in gen_params.items():
            if fp_type not in self.supported_generators:
                raise KeyError(
                    f"Invalid fingerprint `{fp_type}`. Supported fingerprints: {self.supported_generators}"
                )
            supported_params = list(self._generator_params[fp_type].keys())
            for param in fp_params:
                if param not in supported_params:
                    raise KeyError(
                        f"Invalid parameter `{param}` for `{fp_type}` generator. Supported parameters: {', '.join(supported_params)}"
                    )

    def set_tqdm(self, tqdm_struct, tqdm_fp, tqdm_rdkit) -> None:
        """
        Set connection to the tqdm widgets showing 2D
        structure and fingerprint generation progress.
        """

        self._tqdm_struct = tqdm_struct
        self._tqdm_fp = tqdm_fp
        self._tqdm_rdkit = tqdm_rdkit

    def sample_to_df(
        self, sample: list[dict] | pd.DataFrame, smiles_col
    ) -> pd.DataFrame:
        """
        Convert a list or DataFrame of molecule records to a DataFrame
        with molecular fingerprints and 2D structures.

        Args:
            sample (list[dict]): Molecule documents.

        Returns:
            pd.DataFrame: Fully enriched DataFrame with images and fingerprints.
        """

        if not isinstance(sample, pd.DataFrame):
            if not sample:
                return pd.DataFrame()
            df = pd.DataFrame(sample)
        elif sample.empty:
            return pd.DataFrame()
        else:
            df = sample

        df = self._generate_images_parallel(df, smiles_col)
        df = self._add_fingerprints(df, smiles_col)

        # Drop rows where any fingerprint failed
        df.dropna(subset=self.supported_generators, inplace=True)

        return df

    def single_sample_to_df(self, sample: dict, smiles_col) -> pd.DataFrame:
        """
        Convert a single molecule record to a DataFrame.

        Args:
            sample (dict): Single molecule document.

        Returns:
            pd.DataFrame: Single-row DataFrame with image and fingerprints.
        """

        df = pd.DataFrame(sample)
        smiles = df[smiles_col].iloc[0]
        df["img"] = self._smiles_to_img_path(smiles)
        df = self._add_fingerprints(df, smiles_col)

        return df
    
    def mol_to_img(self, doc: dict, smiles_col: str) -> str:
        smiles = doc[smiles_col]
        return self._smiles_to_img_path(smiles)
    
    def generate_rdkit_features(
            self,
            df: pd.DataFrame,
            SMARTS: dict[str, str],
            smiles_column: str = "smiles",
            max_workers: int = 4
        ):

        smiles_list = df[smiles_column].tolist()

        smiles_iter = self._tqdm_rdkit(
            smiles_list,
            desc="Generating features...",
            total=len(smiles_list),
            leave=True,
            colour="#666666",
            mininterval=0.1,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            features = list(executor.map(partial(self._smiles_to_features, SMARTS=SMARTS), smiles_iter))

        df = df.join(pd.DataFrame(features))
        return df

    def _create_generator_defaults(self) -> dict[str, dict[str, int | bool]]:
        """
        Create a dictionary of default values for all supported arguments
        of type `int` and `bool` for all supported fingerprint generators.

        Returns:
            dict[str, dict[str, int | bool]]: The dictionary of default values.
        """

        generator_params = {}
        for name, gen in self._supported_generators.items():

            if name == "ATMOMACCS":
                generator_params[name] = {"bit_width": 6}
                continue

            generator_params[name] = {}
            args_dict = self._parse_signature(gen)

            if not args_dict:
                # fallback default for fpSize
                generator_params[name]["fpSize"] = 2048
                continue

            for arg, arg_info in args_dict.items():
                arg_type = arg_info.get("type")
                default = arg_info.get("default")

                if arg_type == "int" and default is not None:
                    try:
                        generator_params[name][arg] = int(default)
                    except ValueError:
                        continue  # skip invalid int defaults
                elif arg_type == "bool" and default is not None:
                    generator_params[name][arg] = str(default).lower() == "true"

        return generator_params

    def _parse_signature(self, generator) -> dict[str, dict[str, Any]]:
        """
        Parse rdkit fingerprint generator function signature into a dict of parameters.

        Returns:
            dict[str, dict[str, Any]]: Dictionary or supported parameters for the generator
                including the type and default value.
        """

        signature = generator.__doc__
        match = re.search(r"\(([\s\S]*?)\)\s*->", signature)
        if not match:
            return {}
        args_block = match.group(1)

        # Remove optional-bracket syntax "[ ... ]"
        args_block = args_block.replace("[", "").replace("]", "")

        # Split on commas not inside parentheses
        raw_args = [arg.strip() for arg in args_block.split(",") if arg.strip()]

        args_dict = {}
        for arg in raw_args:
            # Match each parameter e.g. (int)radius=3
            m = re.match(r"\(([^)]+)\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,]+)", arg)
            if m:
                arg_type, name, default = m.groups()
                args_dict[name] = {"type": arg_type.strip(), "default": default.strip()}
            else:
                # Handle positional args without default: (int)x
                m = re.match(r"\(([^)]+)\)\s*([A-Za-z_][A-Za-z0-9_]*)", arg)
                if m:
                    arg_type, name = m.groups()
                    args_dict[name] = {"type": arg_type.strip(), "default": None}

        return args_dict

    def _generate_images_parallel(
        self, df: pd.DataFrame, smiles_col, max_workers: int = 4
    ) -> pd.DataFrame:
        """
        Generate molecular images in parallel and add their file paths to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with 'smiles' column.

        Returns:
            pd.DataFrame: Modified DataFrame with 'img' column.
        """

        smiles_iter = self._tqdm_struct(
            df[smiles_col],
            desc="Generating 2D structures...",
            total=len(df),
            leave=True,
            colour="#666666",
            mininterval=0.1,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            paths = list(executor.map(self._smiles_to_img_path, smiles_iter))

        df["img"] = paths
        return df

    @staticmethod
    def _smiles_to_img_path(
        smiles: str, out_dir: str = "phiplot/assets/mol_structures"
    ) -> str | None:
        """
        Convert a SMILES string to a 2D structure image and cache the result.

        Args:
            smiles (str): SMILES representation of the molecule.
            out_dir (str): Directory where images will be saved.

        Returns:
            str or None: Path to the saved image, or None if SMILES is invalid.
        """

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        os.makedirs(out_dir, exist_ok=True)
        filename = f"{hashlib.md5(smiles.encode()).hexdigest()}.jpg"
        path = os.path.join(out_dir, filename)

        if not os.path.isfile(path):
            img = Draw.MolToImage(mol, size=(200, 200))
            img.save(path, format="JPEG", quality=70)

        return path

    def _add_fingerprints(
        self, df: pd.DataFrame, smiles_column: str = "smiles"
    ) -> pd.DataFrame:
        """
        Compute and attach all fingerprint types to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with SMILES.
            smiles_column (str): Column containing SMILES.

        Returns:
            pd.DataFrame: DataFrame with new fingerprint columns.
        """

        smiles_list = df[smiles_column].tolist()

        smiles_iter = self._tqdm_fp(
            smiles_list,
            desc="Computing fingerprints...",
            total=len(smiles_list),
            leave=True,
            colour="#666666",
            mininterval=0.1,
        )

        all_fps = MoleculeHandler._compute_fps_parallel(
            smiles_iter, self.supported_generators, self._generator_params
        )
        for fp_type in self.supported_generators:
            df[fp_type] = [entry.get(fp_type) if entry else None for entry in all_fps]

        return df

    @staticmethod
    def _compute_fps_parallel(
        smiles_iter,
        supported_gens: list[str],
        generator_params: dict[str, dict],
        max_workers: int = 4,
    ) -> list[dict[str, np.ndarray]]:
        """
        Compute fingerprints for a list of SMILES strings in parallel using multiple processes.

        Args:
            smiles_iter: Tqdm iterable containing the list of SMILES strings to process.
            supported_gens (list[str]): List of supported generators.
            generator_params (dict[str, dict]): The parameters for each fingerprint generator.
            max_workers (int, optional): Number of worker processes to use. Defaults to 4.

        Returns:
            list[dict[str, np.ndarray]]:
                List of dictionaries, each containing fingerprints for one molecule.
                Entries are None for SMILES strings that failed to parse.
        """

        compute = partial(
            MoleculeHandler._compute_all_fps,
            supported_gens=supported_gens,
            generator_params=generator_params,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compute, smiles_iter))
        return results

    @staticmethod
    def _compute_all_fps(
        smiles: str, supported_gens: list[str], generator_params: dict[str, dict]
    ) -> dict[str, np.ndarray] | None:
        """
        Compute all supported fingerprints for a given SMILES string.

        Args:
            smiles (str): SMILES representation of the molecule.
            supported_gens (list[str]): List of supported generators.
            generator_params (dict[str, dict]): The parameters for each fingerprint generator.

        Returns:
            dict[str, np.ndarray] | None:
                Dictionary mapping fingerprint type names to binary numpy arrays.
                Returns None if the SMILES is invalid or cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        result = {}
        for fp_type in supported_gens:
            try:
                generator = MoleculeHandler._create_generator(
                    fp_type, **generator_params[fp_type]
                )
                fp = generator.GetFingerprint(mol)
                if fp_type == "ATMOMACCS":
                    result[fp_type] = fp
                else:
                    result[fp_type] = np.array([int(b) for b in fp.ToBitString()])
            except Exception as e:
                logger.warning(
                    f"Failed to compute {fp_type} fingerprint for {smiles}: {e}"
                )
                result[fp_type] = None
        return result

    @staticmethod
    def _create_generator(
        fp_type: str, **kwargs
    ) -> rdFingerprintGenerator.FingerprintGenerator32:
        """
        Create an RDKit fingerprint generator of a specified type with specified parameters.

        Args:
            fp_type (str): Type of fingerprint to generate.
            **kwargs: Parameters controlling the fingerprint generation.

        Returns:
            rdkit.Chem.rdFingerprintGenerator.FingerprintGenerator:
                A configured fingerprint generator instance.

        Raises:
            ValueError: If an unknown fingerprint type is provided.
        """

        dispatch = {
            "Morgan": lambda: rdFingerprintGenerator.GetMorganGenerator(**kwargs),
            "RDKit": lambda: rdFingerprintGenerator.GetRDKitFPGenerator(**kwargs),
            "AtomPairs": lambda: rdFingerprintGenerator.GetAtomPairGenerator(**kwargs),
            "TopologicalTorsions": lambda: rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                **kwargs
            ),
            "ATMOMACCS": lambda: ATMOMACCSGenerator(**kwargs),
        }

        try:
            return dispatch[fp_type]()
        except KeyError:
            raise ValueError(
                f"Unknown fingerprint type: {fp_type}. Supported: {list(dispatch)}"
            )

    @staticmethod
    def _smiles_to_features(smiles: str, SMARTS: dict) -> dict | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        result = dict()
        for name, pattern in SMARTS.items():
            query = Chem.MolFromSmarts(pattern)
            if query is None:
                logger.error(f"Invalid SMARTS pattern: {pattern}")
            matches = mol.GetSubstructMatches(query)
            result[name] = len(matches)

        return result