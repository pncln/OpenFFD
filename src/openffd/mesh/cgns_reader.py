"""CGNS mesh loading helpers.

CGNS files may use either HDF5 or the older ADF container.  ``meshio`` reads
the HDF5 variant, while VTK supports both variants and exposes CGNS zones as
multiblock datasets.  This module normalizes either representation to the
``meshio.Mesh`` object used by the rest of OpenFFD.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def read_cgns_mesh(filename: str) -> Any:
    """Read an HDF5- or ADF-backed CGNS mesh.

    The native meshio reader is preferred for HDF5 CGNS files.  VTK is used
    as a fallback because it can also read legacy ADF CGNS files.
    """
    import meshio

    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError("CGNS mesh file not found: {}".format(filename))

    meshio_error = None
    try:
        mesh = meshio.read(str(path), file_format="cgns")
        if len(mesh.points) > 0:
            return mesh
    except Exception as exc:
        meshio_error = exc
        logger.debug("meshio could not read CGNS file %s: %s", path, exc)

    try:
        import pyvista as pv

        vtk_mesh = pv.read(str(path))
        mesh = _pyvista_to_meshio(vtk_mesh)
    except Exception as exc:
        details = "meshio: {}; VTK: {}".format(meshio_error, exc)
        raise ValueError(
            "Failed to read CGNS mesh '{}'. The file may be invalid or the "
            "installed VTK build may lack CGNS support. {}".format(path, details)
        ) from exc

    if len(mesh.points) == 0:
        raise ValueError("CGNS mesh '{}' contains no points".format(path))

    logger.info(
        "Read CGNS mesh with VTK: %d points, %d cells, zones=%s",
        len(mesh.points),
        sum(len(block.data) for block in mesh.cells),
        list(mesh.cell_sets),
    )
    return mesh


def _iter_datasets(
    data: Any, path: Tuple[str, ...] = ()
) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    """Yield leaf PyVista datasets and their multiblock paths."""
    import pyvista as pv

    if isinstance(data, pv.MultiBlock):
        for index in range(data.n_blocks):
            block = data[index]
            if block is None:
                continue
            name = data.get_block_name(index) or "block_{}".format(index)
            yield from _iter_datasets(block, path + (str(name),))
    elif isinstance(data, pv.DataSet):
        yield path, data


def _zone_candidate(path: Sequence[str]) -> str:
    """Derive a useful zone name from VTK's CGNS multiblock hierarchy."""
    if not path:
        return "default"
    if path[-1].lower() == "internal" and len(path) > 1:
        return path[-2]
    return path[-1]


def _unique_zone_names(paths: Sequence[Tuple[str, ...]]) -> List[str]:
    candidates = [_zone_candidate(path) for path in paths]
    counts = Counter(candidates)
    names = []
    used = set()

    for path, candidate in zip(paths, candidates):
        name = candidate if counts[candidate] == 1 else "/".join(path)
        base_name = name
        suffix = 2
        while name in used:
            name = "{}_{}".format(base_name, suffix)
            suffix += 1
        names.append(name)
        used.add(name)

    return names


def _dataset_cells(dataset: Any) -> List[Any]:
    """Convert one PyVista dataset's VTK connectivity to meshio blocks."""
    from meshio._vtk_common import vtk_cells_from_data

    grid = dataset.cast_to_unstructured_grid()
    if grid.n_cells == 0:
        return []

    cells, _ = vtk_cells_from_data(
        np.asarray(grid.cell_connectivity),
        np.asarray(grid.offset[1:]),
        np.asarray(grid.celltypes),
        {},
    )
    return cells


def _pyvista_to_meshio(data: Any) -> Any:
    """Flatten a PyVista CGNS multiblock dataset into a meshio mesh."""
    import meshio

    leaves = list(_iter_datasets(data))
    if not leaves:
        raise ValueError("VTK returned no datasets for the CGNS file")

    zone_names = _unique_zone_names([path for path, _ in leaves])
    point_arrays = []
    cell_blocks = []
    cell_block_zones = []
    point_offset = 0

    for zone_name, (_, dataset) in zip(zone_names, leaves):
        points = np.asarray(dataset.points)
        if len(points) == 0:
            continue

        point_arrays.append(points)
        try:
            blocks = _dataset_cells(dataset)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Skipping unsupported cells in CGNS zone '%s': %s",
                zone_name,
                exc,
            )
            blocks = []

        for block in blocks:
            connectivity = np.asarray(block.data, dtype=np.int64) + point_offset
            cell_blocks.append(meshio.CellBlock(block.type, connectivity))
            cell_block_zones.append(zone_name)

        point_offset += len(points)

    if not point_arrays:
        raise ValueError("VTK returned no points for the CGNS file")

    points = np.concatenate(point_arrays, axis=0)
    cell_sets = {}
    for zone_name in dict.fromkeys(cell_block_zones):
        cell_sets[zone_name] = [
            (
                np.arange(len(block.data), dtype=np.int64)
                if block_zone == zone_name
                else np.empty(0, dtype=np.int64)
            )
            for block, block_zone in zip(cell_blocks, cell_block_zones)
        ]

    return meshio.Mesh(points=points, cells=cell_blocks, cell_sets=cell_sets)
