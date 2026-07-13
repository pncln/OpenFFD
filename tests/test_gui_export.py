import numpy as np
import pytest
import vtk

import openffd.gui.main as gui_main
from openffd.io.export import read_ffd_3df, read_ffd_xyz


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(message)


class _FFDPanel:
    def get_ffd_mode(self):
        return "standard"

    def get_control_dimensions(self):
        return (2, 2, 2)


class _Window:
    def __init__(self, control_points):
        self.ffd_control_points = control_points
        self.hierarchical_ffd = None
        self.ffd_panel = _FFDPanel()
        self._status_bar = _StatusBar()

    def statusBar(self):
        return self._status_bar


@pytest.mark.parametrize(
    ("extension", "reader"),
    [(".3df", read_ffd_3df), (".xyz", read_ffd_xyz)],
)
def test_gui_standard_ffd_export(tmp_path, monkeypatch, extension, reader):
    control_points = np.arange(24, dtype=float).reshape(8, 3)
    output_path = tmp_path / ("ffd_box" + extension)
    window = _Window(control_points)

    monkeypatch.setattr(
        gui_main.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output_path), ""),
    )
    monkeypatch.setattr(
        gui_main,
        "show_error_dialog",
        lambda *args, **kwargs: pytest.fail("GUI export reported an error"),
    )

    gui_main.OpenFFDMainWindow.on_export_ffd(window)

    assert output_path.exists()
    np.testing.assert_allclose(reader(str(output_path)), control_points)

    if extension == ".xyz":
        plot3d_reader = vtk.vtkMultiBlockPLOT3DReader()
        plot3d_reader.SetXYZFileName(str(output_path))
        plot3d_reader.SetBinaryFile(False)
        plot3d_reader.SetMultiGrid(True)
        plot3d_reader.Update()

        output = plot3d_reader.GetOutput()
        assert output.GetNumberOfBlocks() == 1
        grid = output.GetBlock(0)
        dimensions = [0, 0, 0]
        grid.GetDimensions(dimensions)
        assert tuple(dimensions) == (2, 2, 2)
        assert grid.GetNumberOfPoints() == len(control_points)
