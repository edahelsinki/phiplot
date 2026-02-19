from .ATMOMACCS_v5 import pyGenATMOKeys


class ATMOMACCSGenerator:
    """
    Wraps the ATMOMACCS fingerprinting algorithm into a
    generator like interface.

    Args:
        bit_width (int): The number of bits to allocate for
            binary encoding carbons and oxygens.
        version (int): The version of the generator in the range 1-5.
    """

    def __init__(self, bit_width: int, version: int = 4):
        self.bit_width = bit_width
        self.version = version

    def GetFingerprint(self, mol):
        return pyGenATMOKeys(mol, version=self.version, bit_width=self.bit_width)
