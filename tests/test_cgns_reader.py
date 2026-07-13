from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from openffd.gui.visualization import _prepare_surface_mesh_for_rendering
from openffd.mesh.general import extract_patch_points, read_general_mesh
from openffd.mesh.zone_extractor import ZoneExtractor


CGNS_FIXTURE = (
    Path(__file__).resolve().parents[1] / "examples" / "cgns_meshes" / "mesh006.cgns"
)


@pytest.mark.skipif(
    not CGNS_FIXTURE.exists(), reason="CGNS example mesh is unavailable"
)
def test_read_adf_cgns_mesh():
    mesh = read_general_mesh(str(CGNS_FIXTURE))

    assert mesh.points.shape == (2606, 3)
    assert [(block.type, block.data.shape) for block in mesh.cells] == [
        ("hexahedron20", (476, 20))
    ]
    assert "mesh006" in mesh.cell_sets
    np.testing.assert_allclose(
        [mesh.points.min(axis=0), mesh.points.max(axis=0)],
        [[0.0, -100.0, -200.0], [1000.0, 100.0, 200.0]],
        atol=1.0e-10,
    )


@pytest.mark.skipif(
    not CGNS_FIXTURE.exists(), reason="CGNS example mesh is unavailable"
)
def test_extract_cgns_zone_points():
    mesh = read_general_mesh(str(CGNS_FIXTURE))

    points = extract_patch_points(mesh, "mesh006")

    assert points.shape == (2606, 3)


@pytest.mark.skipif(not CGNS_FIXTURE.exists(), reason="CGNS example mesh is unavailable")
def test_cgns_zone_metadata():
    zones = ZoneExtractor(str(CGNS_FIXTURE)).get_zone_info()

    assert zones["mesh006"].cell_count == 476
    assert zones["mesh006"].point_count == 2606
    assert zones["mesh006"].element_types == {"hexahedron20"}


@pytest.mark.skipif(not CGNS_FIXTURE.exists(), reason="CGNS example mesh is unavailable")
def test_cgns_rendering_preserves_quadrilateral_faces():
    mesh = read_general_mesh(str(CGNS_FIXTURE))
    surface = _prepare_surface_mesh_for_rendering(pv.from_meshio(mesh))

    face_sizes = []
    offset = 0
    while offset < len(surface.faces):
        face_size = int(surface.faces[offset])
        face_sizes.append(face_size)
        offset += face_size + 1

    assert surface.n_cells == 430
    assert set(face_sizes) == {4}
