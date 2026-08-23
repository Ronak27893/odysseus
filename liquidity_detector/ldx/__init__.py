"""Manufactured-liquidity-event detector for US small caps.

Hypothesis under test: in ramp-and-dump and dilution-into-strength events the
price move is not the objective -- it is the mechanism for creating enough
depth to exit a position that could not otherwise be sold. This package tests
whether that leaves a separable signature.

Every flag is a signature, not a verdict. See METHODS.md.
"""

__version__ = "0.1.0"
