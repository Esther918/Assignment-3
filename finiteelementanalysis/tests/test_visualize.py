from finiteelementanalysis import visualize as viz
from finiteelementanalysis import pre_process as pre
import numpy as np
import os
from pathlib import Path
import pytest


element_types = ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"]
@pytest.mark.parametrize("ele_type", element_types)
def test_plot_mesh_2D(ele_type):
    x_lower, y_lower, x_upper, y_upper = 0, 0, 10, 5
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Compute displacements: u = 0.1 * x, v = 0.25 * y
    displacements = np.zeros_like(coords)
    displacements[:, 0] = 0.1 * coords[:, 0] + 0.5 * np.random.random(coords[:, 0].shape) # artifical x-displacement
    displacements[:, 1] = 0.25 * coords[:, 1] + 0.5 * np.random.random(coords[:, 1].shape) # artificial y-displacement

    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    fname = "test_plot_mesh_2D_" + ele_type + ".png"
    plot_file = test_dir / fname

    viz.plot_mesh_2D(str(plot_file), ele_type, coords, connect, displacements)

    # Validate the file was created and is not empty
    assert plot_file.exists(), f"Plot file {plot_file} was not created."
    assert os.stat(plot_file).st_size > 0, f"Plot file {plot_file} is empty."


@pytest.mark.parametrize("ele_type", element_types)
def test_make_deformation_gif(ele_type):
    x_lower, y_lower, x_upper, y_upper = 0, 0, 10, 5
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Create synthetic displacements over 5 time steps
    n_nodes = coords.shape[0]
    n_steps = 5
    displacements_all = []
    for t in range(n_steps):
        disp = np.zeros_like(coords)
        disp[:, 0] = (0.1 + 0.02 * t) * coords[:, 0] + 0.1 * np.random.random(n_nodes)
        disp[:, 1] = (0.25 + 0.02 * t) * coords[:, 1] + 0.1 * np.random.random(n_nodes)
        displacements_all.append(disp.flatten())

    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    gif_path = test_dir / f"test_make_deformation_gif_{ele_type}.gif"

    # Run gif generator
    viz.make_deformation_gif(displacements_all, coords, connect, ele_type, gif_path, magnification=1.0)

    # Validate that the gif was created and is non-empty
    assert gif_path.exists(), f"GIF file {gif_path} was not created."
    assert os.stat(gif_path).st_size > 0, f"GIF file {gif_path} is empty."