#!/usr/bin/env python3
"""Test script to debug force coefficient extraction."""

import sys
import os
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from openffd.cfd.solvers.openfoam import OpenFOAMSolver
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create solver instance
solver = OpenFOAMSolver()

# Test force coefficient extraction
case_dir = Path("/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/naca0012_case")
logger.info(f"Testing force extraction from: {case_dir}")

# Test the extraction
force_coeffs = solver._extract_force_coefficients(case_dir)
logger.info(f"Result: {force_coeffs}")

if force_coeffs:
    logger.info(f"Cd: {force_coeffs.cd}")
    logger.info(f"Cl: {force_coeffs.cl}")
    logger.info(f"Cm: {force_coeffs.cm}")
else:
    logger.warning("Force coefficients extraction failed")