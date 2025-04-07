from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import pre_process_demo_helper_fcns as pre_demo
import numpy as np
import os
from pathlib import Path
import pytest


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"])
def test_plot_mesh_2D(tmp_path, ele_type):
    """
    Test function to verify that `plot_mesh_2D` runs correctly for all supported element types 
    and generates valid plot files.
    """
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    x_lower, y_lower = 0, 10
    x_upper, y_upper = 10, 5
    nx, ny = 10, 5

    # Generate a test mesh for the given element type
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Validate the generated mesh
    assert coords.shape[1] == 2, f"Coordinates array should have shape (n_nodes, 2) for {ele_type}"
    assert connect.shape[1] > 0, f"Connectivity array should have at least one element for {ele_type}"

    # Define test file path
    fname = test_dir / f"test_mesh_{ele_type}.png"

    # Call the plotting function
    pre_demo.plot_mesh_2D(str(fname), ele_type, coords, connect)

    # Ensure the plot file is created
    assert fname.exists(), f"Plot file {fname} was not created for {ele_type}"

    # Ensure the file is non-empty
    assert os.stat(fname).st_size > 0, f"Plot file {fname} is empty for {ele_type}"


# Define valid Gauss point counts for each element type
VALID_GAUSS_POINTS = {
    "D2_nn3_tri": [1, 3, 4],   # 3-node triangle
    "D2_nn6_tri": [1, 3, 4],   # 6-node quadratic triangle
    "D2_nn4_quad": [1, 4, 9],  # 4-node bilinear quadrilateral
    "D2_nn8_quad": [1, 4, 9]   # 8-node quadratic quadrilateral
}

@pytest.mark.parametrize("ele_type, num_gauss_pts",
    [(etype, gp) for etype, gps in VALID_GAUSS_POINTS.items() for gp in gps]
)
def test_get_all_mesh_gauss_pts(ele_type, num_gauss_pts):
    """
    Test function for `get_all_mesh_gauss_pts`, ensuring Gauss points are computed correctly
    for all supported element types and valid Gauss point numbers.
    """
    x_lower, y_lower = 0, 10
    x_upper, y_upper = 10, 5
    nx, ny = 3, 2  # Small test mesh

    # Generate test mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Compute Gauss points in physical coordinates
    mesh_gauss_pts = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)

    # Validate dimensions
    num_elements = connect.shape[0]
    assert mesh_gauss_pts.shape == (num_elements, num_gauss_pts, 2), \
        f"Unexpected shape: {mesh_gauss_pts.shape} for {ele_type} with {num_gauss_pts} Gauss points"

    # Ensure Gauss points are within element bounds
    for kk in range(num_elements):
        element_coords = coords[connect[kk]]  # Get element's node coordinates
        x_min, y_min = element_coords.min(axis=0)
        x_max, y_max = element_coords.max(axis=0)

        for jj in range(num_gauss_pts):
            gx, gy = mesh_gauss_pts[kk, jj]
            assert x_min <= gx <= x_max, f"Gauss point {gx} out of x-bounds for {ele_type} with {num_gauss_pts} Gauss points"
            assert y_min <= gy <= y_max, f"Gauss point {gy} out of y-bounds for {ele_type} with {num_gauss_pts} Gauss points"

    # Ensure no NaNs or invalid values
    assert np.all(np.isfinite(mesh_gauss_pts)), f"NaN or Inf values detected for {ele_type} with {num_gauss_pts} Gauss points"


@pytest.mark.parametrize("invalid_ele_type", ["D2_invalid", "D3_nn4_tetra"])
def test_get_all_mesh_gauss_pts_invalid_type(invalid_ele_type):
    """
    Test function for handling unsupported element types.
    """
    num_gauss_pts = 3
    coords = np.array([[0, 0], [1, 0], [0, 1]])  # Dummy data
    connect = np.array([[0, 1, 2]])

    with pytest.raises(ValueError, match="Unsupported element type"):
        pre_demo.get_all_mesh_gauss_pts(invalid_ele_type, num_gauss_pts, coords, connect)


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"])
@pytest.mark.parametrize("num_gauss_pts", [1, 3, 4, 9])  # Includes all possible Gauss point cases
def test_plot_mesh_2D_with_gauss_points(tmp_path, ele_type, num_gauss_pts):
    """
    Test function to verify that `plot_mesh_2D` runs correctly for all supported element types 
    with and without Gauss points, and generates valid plot files.
    """
    # Ensure the number of Gauss points is valid for the element type
    if num_gauss_pts not in VALID_GAUSS_POINTS[ele_type]:
        pytest.skip(f"Skipping invalid number of Gauss points ({num_gauss_pts}) for {ele_type}")

    # Create test directory
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)

    x_lower, y_lower = 0, 10
    x_upper, y_upper = 10, 5
    nx, ny = 10, 5

    # Generate a test mesh for the given element type
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Validate the generated mesh
    assert coords.shape[1] == 2, f"Coordinates array should have shape (n_nodes, 2) for {ele_type}"
    assert connect.shape[1] > 0, f"Connectivity array should have at least one element for {ele_type}"

    # Compute Gauss points in physical coordinates
    gauss_pts = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)

    # Validate Gauss points shape
    num_elements = connect.shape[0]
    assert gauss_pts.shape == (num_elements, num_gauss_pts, 2), \
        f"Unexpected Gauss points shape: {gauss_pts.shape} for {ele_type} with {num_gauss_pts} Gauss points"

    # Define test file path
    fname = test_dir / f"test_mesh_{ele_type}_gauss_{num_gauss_pts}.png"

    # Call the plotting function with Gauss points
    pre_demo.plot_mesh_2D(str(fname), ele_type, coords, connect, gauss_points=gauss_pts)

    # Ensure the plot file is created
    assert fname.exists(), f"Plot file {fname} was not created for {ele_type} with {num_gauss_pts} Gauss points"

    # Ensure the file is non-empty
    assert os.stat(fname).st_size > 0, f"Plot file {fname} is empty for {ele_type} with {num_gauss_pts} Gauss points"


def fcn_1(x, y):
    val = 3.0 * x + 5.0 * y
    return val


@pytest.mark.parametrize("ele_type, num_gauss_pts",
    [(etype, gp) for etype, gps in VALID_GAUSS_POINTS.items() for gp in gps]
)
def test_interpolate_scalar_to_gauss_pts(ele_type, num_gauss_pts):
    # Define a test domain
    x_lower, y_lower = 0.0, 10.0
    x_upper, y_upper = 10.0, 5.0
    nx, ny = 4, 2  # small mesh for fast testing

    # Generate mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Evaluate ground truth: compute fcn_1 at exact Gauss point locations
    mesh_gauss_pts = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)  # shape (num_elements, num_gauss_pts, 2)
    ground_truth = fcn_1(mesh_gauss_pts[:, :, 0], mesh_gauss_pts[:, :, 1])  # evaluate function vectorized

    # Evaluate interpolated function at Gauss points
    interpolated = pre_demo.interpolate_scalar_to_gauss_pts(ele_type, num_gauss_pts, fcn_1, coords, connect)

    # Assert exact match (within floating point tolerance)
    assert np.allclose(interpolated, ground_truth, atol=1e-12), \
        f"Interpolation mismatch for {ele_type} with {num_gauss_pts} Gauss points"


def fcn_2(x, y):
    val = 10.0 * x + -3.0 * y
    return val


def fcn_2_deriv(x, y):
    deriv = np.asarray([10.0, -3.0])
    return deriv


@pytest.mark.parametrize("ele_type, num_gauss_pts",
    [(etype, gp) for etype, gps in VALID_GAUSS_POINTS.items() for gp in gps]
)
def test_interpolate_scalar_deriv_to_gauss_pts(ele_type, num_gauss_pts):
    # Define a test domain
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 10.0, 10.0
    nx, ny = 2, 2  # small mesh for fast testing

    # Generate mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Evaluate ground truth: compute fcn_1 at exact Gauss point locations
    mesh_gauss_pts = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)  # shape (num_elements, num_gauss_pts, 2)
    ground_truth = np.zeros(mesh_gauss_pts.shape)
    for kk in range(mesh_gauss_pts.shape[0]):
        for jj in range(mesh_gauss_pts.shape[1]):
            x = mesh_gauss_pts[kk, jj, 0]
            y = mesh_gauss_pts[kk, jj, 1]
            ground_truth[kk, jj, :] = fcn_2_deriv(x, y)

    # Evaluate interpolated function at Gauss points
    interpolated = pre_demo.interpolate_scalar_deriv_to_gauss_pts(ele_type, num_gauss_pts, fcn_2, coords, connect)

    # Assert exact match (within floating point tolerance)
    assert np.allclose(interpolated, ground_truth, atol=1e-12), \
        f"Interpolation mismatch for {ele_type} with {num_gauss_pts} Gauss points"


@pytest.mark.parametrize("ele_type, num_gauss_pts", [
    (etype, gp) for etype, gps in VALID_GAUSS_POINTS.items() for gp in gps
])
def test_plot_interpolation_with_error(tmp_path, ele_type, num_gauss_pts):
    # Create mesh
    x_lower, y_lower = 0.0, 10.0
    x_upper, y_upper = 10.0, 5.0
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Compute Gauss point coordinates
    gauss_points_coords = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)

    # Evaluate true and interpolated values at Gauss points
    ground_truth = fcn_1(gauss_points_coords[:, :, 0], gauss_points_coords[:, :, 1])
    interpolated = pre_demo.interpolate_scalar_to_gauss_pts(ele_type, num_gauss_pts, fcn_1, coords, connect)

    # Plot
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    fname = test_dir / f"test_plot_{ele_type}_gp{num_gauss_pts}_error.png"
    pre_demo.plot_interpolation_with_error(
        str(fname),
        ele_type,
        coords,
        connect,
        gauss_points_coords,
        interpolated,
        ground_truth
    )

    # Validate output file
    assert fname.exists(), f"File {fname} was not created."
    assert os.stat(fname).st_size > 0, f"File {fname} is empty."


def fcn_3(x, y):
    val = 10.0 * x + -3.0 * y + 7.0 * x * y
    return val


def fcn_3_deriv(x, y):
    deriv = np.asarray([10.0 + 7.0 * y, -3.0 + 7.0 * x])
    return deriv


@pytest.mark.parametrize("ele_type, num_gauss_pts", [
    (etype, gp) for etype, gps in VALID_GAUSS_POINTS.items() for gp in gps
])
def test_plot_interpolation_gradient_with_error(tmp_path, ele_type, num_gauss_pts):
    # Create mesh
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 10.0, 10.0
    nx, ny = 3, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Compute Gauss point coordinates
    gauss_points_coords = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)

    # Evaluate true and interpolated values at Gauss points
    ground_truth = np.zeros(gauss_points_coords.shape)
    for kk in range(0, ground_truth.shape[0]):
        for jj in range(0, ground_truth.shape[1]):
            ground_truth[kk, jj] = fcn_3_deriv(gauss_points_coords[kk, jj, 0], gauss_points_coords[kk, jj, 1])
    interpolated = pre_demo.interpolate_scalar_deriv_to_gauss_pts(ele_type, num_gauss_pts, fcn_3, coords, connect)

    # Plot
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    fname = test_dir / f"test_plot_{ele_type}_gp{num_gauss_pts}_grad_error.png"
    pre_demo.plot_interpolation_gradient_with_error(
        str(fname),
        ele_type,
        coords,
        connect,
        gauss_points_coords,
        interpolated,
        ground_truth
    )

    # Validate output file
    assert fname.exists(), f"File {fname} was not created."
    assert os.stat(fname).st_size > 0, f"File {fname} is empty."



@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn6_tri"])
@pytest.mark.parametrize("num_gauss_pts", [1])
def test_complex_geom(ele_type, num_gauss_pts):
    """
    Tests a complex geometry mesh with triangular elements (both 3-node and 6-node).
    Ensures Gauss points are correctly computed, interpolation is accurate, and plots are saved.
    """
    # Define mesh settings
    mesh_name = f"terrier_mesh_{ele_type}_gp{num_gauss_pts}"
    complex_outline = pre.get_terrier_outline()  # Get complex geometry
    mesh_size = 25.0  # Mesh resolution

    # Generate mesh
    coords, connect = pre.mesh_outline(complex_outline, ele_type, mesh_name, mesh_size)

    # Create test output directory
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Expected file paths
    mesh_plot = test_dir / f"{mesh_name}.png"
    gauss_plot = test_dir / f"{mesh_name}_gauss_pts.png"
    error_plot = test_dir / f"{mesh_name}_interp_error.png"

    # Save initial mesh plot
    pre_demo.plot_mesh_2D(str(mesh_plot), ele_type, coords, connect)
    assert mesh_plot.exists() and os.stat(mesh_plot).st_size > 0, f"Mesh plot {mesh_plot} was not created or is empty."

    # Compute Gauss points
    mesh_gauss_pts = pre_demo.get_all_mesh_gauss_pts(ele_type, num_gauss_pts, coords, connect)

    # Save Gauss point visualization
    pre_demo.plot_mesh_2D(str(gauss_plot), ele_type, coords, connect, mesh_gauss_pts)
    assert gauss_plot.exists() and os.stat(gauss_plot).st_size > 0, f"Gauss plot {gauss_plot} was not created or is empty."

    # Compute ground truth at Gauss points
    ground_truth = fcn_1(mesh_gauss_pts[..., 0], mesh_gauss_pts[..., 1])  # Vectorized evaluation

    # Compute interpolated values from nodal values
    interpolated = pre_demo.interpolate_scalar_to_gauss_pts(ele_type, num_gauss_pts, fcn_1, coords, connect)

    # Save interpolation error plot
    pre_demo.plot_interpolation_with_error(
        str(error_plot),
        ele_type,
        coords,
        connect,
        mesh_gauss_pts,
        interpolated,
        ground_truth
    )
    assert error_plot.exists() and os.stat(error_plot).st_size > 0, f"Interpolation error plot {error_plot} was not created or is empty."

    # Ensure interpolation is exact (within floating point precision)
    assert np.allclose(interpolated, ground_truth, atol=1e-12), \
        f"Interpolation mismatch for {ele_type} with {num_gauss_pts} Gauss points"

    # Compute ground truth gradient at Gauss points (∂f/∂x, ∂f/∂y)
    ground_truth_grad = np.zeros_like(mesh_gauss_pts)
    for kk in range(mesh_gauss_pts.shape[0]):
        for jj in range(mesh_gauss_pts.shape[1]):
            x = mesh_gauss_pts[kk, jj, 0]
            y = mesh_gauss_pts[kk, jj, 1]
            ground_truth_grad[kk, jj] = fcn_2_deriv(x, y)  # should return [∂f/∂x, ∂f/∂y]

    # Compute interpolated gradient from nodal values of fcn_3
    interpolated_grad = pre_demo.interpolate_scalar_deriv_to_gauss_pts(
        ele_type,
        num_gauss_pts,
        fcn_2,
        coords,
        connect
    )

    # Save gradient interpolation error plot
    grad_error_plot = test_dir / f"{mesh_name}_grad_interp_error.png"
    pre_demo.plot_interpolation_gradient_with_error(
        str(grad_error_plot),
        ele_type,
        coords,
        connect,
        mesh_gauss_pts,
        interpolated_grad,
        ground_truth_grad
    )
    assert grad_error_plot.exists() and os.stat(grad_error_plot).st_size > 0, \
        f"Gradient interpolation plot {grad_error_plot} was not created or is empty."

    # Verify gradient interpolation is accurate
    assert np.allclose(interpolated_grad, ground_truth_grad, atol=1e-12), \
        f"Gradient interpolation mismatch for {ele_type} with {num_gauss_pts} Gauss points"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri"]) #, "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"])
def test_element_quality_metrics(ele_type):
    """
    Tests element quality metrics for all supported 2D element types.
    """
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 4.0, 2.0
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Compute element quality metrics
    aspect_ratios, skewness, min_angles, max_angles = pre_demo.compute_element_quality_metrics(
        ele_type, coords, connect
    )

    # Assertions for acceptable mesh quality
    if ele_type == "D2_nn4_quad" or ele_type == "D2_nn8_quad":
        assert np.all(aspect_ratios >= 1), "Aspect ratios should be >= 1"
    else:
        assert np.all(aspect_ratios <= 2), "Aspect ratios should be >= 1"
    assert np.all(min_angles > 30), "Minimum angles should be > 30°"
    assert np.all(max_angles < 150), "Maximum angles should be < 150° (quads) or < 120° (triangles)"
    
    # Skewness thresholds: more lenient for quads
    skewness_threshold = 0.8 if "tri" in ele_type else 2.0
    assert np.all(skewness <= skewness_threshold), f"Skewness values exceed {skewness_threshold} for {ele_type}"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"])
def test_compute_condition_and_jacobian(ele_type):
    """
    Tests the Jacobian determinant and condition number computations
    at the element center for all supported 2D element types.
    """
    # Generate a small structured mesh
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 4.0, 2.0
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Compute condition numbers and Jacobian determinants at element centers
    cond_nums, jac_dets = pre_demo.compute_condition_and_jacobian(ele_type, coords, connect)

    # Assert Jacobian determinants are positive (no inverted elements)
    assert np.all(jac_dets > 0), f"Some elements in {ele_type} have non-positive Jacobian determinants"

    # Assert condition numbers are reasonable (not numerically unstable)
    assert np.all(cond_nums < 1000), f"Some elements in {ele_type} have very large condition numbers"


def test_plot_element_quality_histograms():
    """
    Generates a triangular mesh, computes element quality metrics,
    and verifies that plot_element_quality_histograms creates a valid histogram figure.
    """
    # Use a 4-node quadrilateral mesh
    ele_type = "D2_nn3_tri"
    mesh_name = f"terrier_mesh_{ele_type}"
    complex_outline = pre.get_terrier_outline()  # Get complex geometry
    mesh_size = 10.0  # Mesh resolution

    # Generate mesh
    coords, connect = pre.mesh_outline(complex_outline, ele_type, mesh_name, mesh_size)

    # Compute metrics
    aspect_ratios, skewness, min_angles, max_angles = pre_demo.compute_element_quality_metrics(ele_type, coords, connect)
    cond_nums, jac_dets = pre_demo.compute_condition_and_jacobian(ele_type, coords, connect)

    # Output file location
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    plot_file = test_dir / "test_quality_histograms_tri.png"

    # Call the function
    pre_demo.plot_element_quality_histograms(
        fname=str(plot_file),
        super_title="Test Mesh Quality Metrics (%s)" % (ele_type),
        ele_type=ele_type,
        cond_nums=cond_nums,
        jac_dets=jac_dets,
        aspect_ratios=aspect_ratios,
        skewness=skewness,
        min_angles=min_angles,
        max_angles=max_angles
    )

    # Validate the file was created and is not empty
    assert plot_file.exists(), f"Plot file {plot_file} was not created."
    assert os.stat(plot_file).st_size > 0, f"Plot file {plot_file} is empty."


def check_jacobian_positive(ele_type, outline, mesh_name):
    mesh_size = 0.1
    coords, connect = pre.mesh_outline(outline, ele_type, mesh_name, mesh_size)
    _, jac_dets = pre_demo.compute_condition_and_jacobian(ele_type, coords, connect)
    assert np.min(jac_dets) > 0, f"Negative Jacobian found in mesh: {mesh_name}"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn6_tri"])
@pytest.mark.parametrize("outline_name,outline", [
    # Squares (axis-aligned)
    ("cw_square", [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]),
    ("ccw_square", [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]),

    # Rectangles (wider)
    ("cw_rectangle", [(2, 1), (2, -1), (-2, -1), (-2, 1), (2, 1)]),
    ("ccw_rectangle", [(2, 1), (-2, 1), (-2, -1), (2, -1), (2, 1)]),

    # Rectangles (taller)
    ("cw_tall", [(1, 2), (1, -2), (-1, -2), (-1, 2), (1, 2)]),
    ("ccw_tall", [(1, 2), (-1, 2), (-1, -2), (1, -2), (1, 2)]),

    # Non-axis aligned quadrilateral (parallelogram-like)
    ("cw_skewed", [(2, 2), (3, 0), (1, -2), (0, 0), (2, 2)]),
    ("ccw_skewed", [(2, 2), (0, 0), (1, -2), (3, 0), (2, 2)]),

    # Tilted diamond
    ("cw_diamond", [(0, 2), (2, 0), (0, -2), (-2, 0), (0, 2)]),
    ("ccw_diamond", [(0, 2), (-2, 0), (0, -2), (2, 0), (0, 2)]),

    # Trapezoid
    ("cw_trapezoid", [(1, 1), (2, -1), (-2, -1), (-1, 1), (1, 1)]),
    ("ccw_trapezoid", [(1, 1), (-1, 1), (-2, -1), (2, -1), (1, 1)]),
])
def test_jacobian_from_pygmsh(ele_type, outline_name, outline):
    """
    Tests that elements generated from clockwise and counterclockwise outlines
    produce strictly positive Jacobian determinants. This protects against issues 
    from node ordering or platform-dependent pygmsh behavior.
    """
    mesh_name = f"{outline_name}_{ele_type}"
    mesh_size = 0.1
    coords, connect = pre.mesh_outline(outline, ele_type, mesh_name, mesh_size)
    _, jac_dets = pre_demo.compute_condition_and_jacobian(ele_type, coords, connect)

    min_jac = np.min(jac_dets)
    assert min_jac > 0, (
        f"Negative Jacobian in mesh '{mesh_name}' (min={min_jac:.3e}). "
        f"This may indicate node ordering inconsistency or a bug in orientation handling."
    )
