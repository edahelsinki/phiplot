import logging
import json
from phiplot import ROOT

logger = logging.getLogger(__name__)


class ParamParser:
    def __init__(self, defaults_file: str):
        with open(ROOT / f"assets/defaults/{defaults_file}") as handle:
            dictdump: dict = json.loads(handle.read())
        
        self._supported_algorithms = []
        self._hyperparams = {}
        self._default_hyperparams = {}
        
        for algo in dictdump:
            self._supported_algorithms.append(algo)
            param_info = dictdump[algo]["params"]
            self._default_hyperparams[algo] = param_info
            algo_params = {}
            for param, info in param_info.items():
                algo_params[param] = info["default"]
            self._hyperparams[algo] = algo_params

        self._algorithm: str | None = None

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algo) -> None:
        if algo in self._supported_algorithms:
            self._algorithm = algo
        else:
            logger.error(f"Unsupported algorithm: {algo}, should be one of: {self._supported_algorithms}")

    @property
    def default_hyperparams(self) -> dict:
        return self._default_hyperparams.copy()

    @property
    def supported_algorithms(self) -> list:
        return self._supported_algorithms.copy()