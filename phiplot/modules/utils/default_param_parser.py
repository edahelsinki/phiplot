import logging
import json
from phiplot import ROOT

logger = logging.getLogger(__name__)

class DefaultParamParser:
    def __init__(self, defaults_file: str):
        with open(ROOT / f"assets/defaults/{defaults_file}") as handle:
            dictdump: dict = json.loads(handle.read())

        self._supported = list(dictdump.keys())
        self._n_items = len(self._supported)
        self._info = dictdump
    
    @property
    def supported(self) -> list:
        return self._supported.copy()

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self._n_items:
            k = self._supported[self._i]
            info = self._info[k]
            self._i += 1
            return k, info
        else:
            raise StopIteration