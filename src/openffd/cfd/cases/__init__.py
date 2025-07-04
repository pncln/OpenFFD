"""Case handlers for different CFD optimization types."""

from .base_case import GenericCase
from .airfoil_case import AirfoilCase
from .heat_transfer_case import HeatTransferCase

__all__ = [
    'GenericCase',
    'AirfoilCase', 
    'HeatTransferCase'
]