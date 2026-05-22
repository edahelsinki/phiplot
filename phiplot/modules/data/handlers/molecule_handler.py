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
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MACCSkeys

from phiplot.modules.ATMOMACCS import *
from phiplot.modules.utils.default_param_parser import DefaultParamParser

logger = logging.getLogger(__name__)


class MACCSGenerator:
    """Wrapper to make RDKit's MACCS function compatible with the generator pattern."""
    def __init__(self, **kwargs):
        pass
        
    def GetFingerprint(self, mol):
        return MACCSkeys.GenMACCSKeys(mol)


class MoleculeHandler:
    """
    Handles SMILES-based molecule processing including:
        - Generating 2D structure images
        - Generating molecular fingerprints (The list of all supported
          generators can be accessed via the `supported_generators` property)
        - Storing results into structured DataFrames
    """

    def __init__(self):
        self._fp_param_parser = DefaultParamParser("fingerprinting_hyperparams.json")

        self._tqdm_struct = None
        self._tqdm_fp = None
        self._tqdm_smarts = None

    @property
    def supported_generators(self):
        return self._fp_param_parser.supported

    def set_tqdm(self, tqdm_struct, tqdm_fp, tqdm_smarts) -> None:
        """
        Set connection to the tqdm widgets showing 2D
        structure and fingerprint generation progress.
        """

        self._tqdm_struct = tqdm_struct
        self._tqdm_fp = tqdm_fp
        self._tqdm_smarts = tqdm_smarts

    def featurize_data(
            self, index: list[str] | pd.Series, smiles: list[str] | pd.Series, fp_params: dict[str, dict[str, Any]] | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert a list of SMILES strings to a DataFrame with molecular fingerprints and 2D structures.

        Args:
            index (list[str] | pd.Series): The index to use for the resulting dataframe.
            smiles (list[str] | pd.Series): The SMILES strings to convert.
            fp_params (dict[str, dict[str, Any]] | None): Dictionary or supported parameters 
                for the generator including the type and default value. Defaults to None.

        Returns:
            (tuple[pd.DataFrame pd.DataFrame]: 
                - DataFrame with paths to 2D structure images
                - DataFrame with all generated fingerprints
        """

        df_2d = self._generate_images_parallel(index, smiles)
        df_fp = self._generate_fingerprints(index, smiles, fp_params)

        return df_2d, df_fp

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
    
    def generate_smarts_features(
            self,
            df: pd.DataFrame,
            SMARTS: dict[str, str],
            smiles_column: str = "smiles",
            max_workers: int = 4
        ):

        smiles_list = df[smiles_column].tolist()

        smiles_iter = self._tqdm_smarts(
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

    def _generate_images_parallel(
        self, index: list[str] | pd.Series, smiles: list[str] | pd.Series, max_workers: int = 4
    ) -> pd.DataFrame:
        """
        Generate molecular images in parallel and add their file paths to the DataFrame.

        Args:
            index (list[str] | pd.Series): The index to use for the resulting dataframe.
            smiles (list[str] | pd.Series): The SMILES strings to convert.
            max_workers (int): Maximum number of parallel workers. Defaults to 4.

        Returns:
            pd.DataFrame: Paths to the 2D structure images
        """

        smiles_iter = self._tqdm_struct(
            smiles,
            desc="Generating 2D structures...",
            total=len(index),
            leave=True,
            colour="#666666",
            mininterval=0.1,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            paths = list(executor.map(self._smiles_to_img_path, smiles_iter))

        return pd.DataFrame(dict(index=index, img=paths))

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
        
        filename = f"{hashlib.md5(smiles.encode()).hexdigest()}.png"
        path = os.path.join(out_dir, filename)

        if not os.path.isfile(path):
            d2d = rdMolDraw2D.MolDraw2DCairo(600, 600)
            
            opts = d2d.drawOptions()
            opts.bondLineWidth = 2 
            opts.addStereoAnnotation = True
            opts.minFontSize = 14 
            opts.maxFontSize = 24 
            opts.padding = 0.1
            
            rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol)
            d2d.FinishDrawing()

            with open(path, "wb") as f:
                f.write(d2d.GetDrawingText())

        return path

    def _generate_fingerprints(
        self, index: list[str] | pd.Series, smiles: list[str] | pd.Series, fp_params: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compute and attach all fingerprint types to the DataFrame.

        Args:
            index (list[str] | pd.Series): The index to use for the resulting dataframe.
            smiles (list[str] | pd.Series): The SMILES strings to convert.
            fp_params (dict[str, dict[str, Any]]): Dictionary or supported 
                parameters for the generator including the type and default value.

        Returns:
            pd.DataFrame: DataFrame with new fingerprint columns.
        """

        smiles_iter = self._tqdm_fp(
            smiles,
            desc="Computing fingerprints...",
            total=len(index),
            leave=True,
            colour="#666666",
            mininterval=0.1,
        )

        all_fps = MoleculeHandler._compute_fps_parallel(
            smiles_iter, self.supported_generators, fp_params
        )
        df = pd.DataFrame(dict(index=index))
        for fp_type in self.supported_generators:
            df[fp_type] = [entry.get(fp_type) if entry else None for entry in all_fps]

        return df

    @staticmethod
    def _compute_fps_parallel(
        smiles_iter,
        supported_gens: list[str],
        fp_params: dict[str, dict],
        max_workers: int = 4,
    ) -> list[dict[str, np.ndarray]]:
        """
        Compute fingerprints for a list of SMILES strings in parallel using multiple processes.

        Args:
            smiles_iter: Tqdm iterable containing the list of SMILES strings to process.
            supported_gens (list[str]): List of supported generators.
            fp_params (dict[str, dict]): The parameters for each fingerprint generator.
            max_workers (int, optional): Number of worker processes to use. Defaults to 4.

        Returns:
            list[dict[str, np.ndarray]]:
                List of dictionaries, each containing fingerprints for one molecule.
                Entries are None for SMILES strings that failed to parse.
        """

        compute = partial(
            MoleculeHandler._compute_all_fps,
            supported_gens=supported_gens,
            fp_params=fp_params,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compute, smiles_iter))
        return results

    @staticmethod
    def _compute_all_fps(
        smiles: str, supported_gens: list[str], fp_params: dict[str, dict]
    ) -> dict[str, np.ndarray] | None:
        """
        Compute all supported fingerprints for a given SMILES string.

        Args:
            smiles (str): SMILES representation of the molecule.
            supported_gens (list[str]): List of supported generators.
            fp_params (dict[str, dict]): The parameters for each fingerprint generator.

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
                    fp_type, **fp_params.get(fp_type, {})
                )
                fp = generator.GetFingerprint(mol)
                if fp_type == "ATMOMACCS":
                    result[fp_type] = fp
                else:
                    result[fp_type] = np.array([int(b) for b in fp.ToBitString()])
            except Exception:
                logger.exception("Error during fingerprinting:")
                logger.warning(
                    f"Failed to compute {fp_type} fingerprint for {smiles}"
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
            "MACCS": lambda: MACCSGenerator(**kwargs)
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