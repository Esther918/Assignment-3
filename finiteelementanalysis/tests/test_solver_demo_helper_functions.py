from finiteelementanalysis import solver_demo_helper_functions as solver_demo
from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import pre_process_demo_helper_fcns as pre_demo
import numpy as np
import os
from pathlib import Path
import pytest
import scipy.sparse as sp
import time


def test_compute_bandwidth_dense():
    K = np.array([
        [10, 2, 0, 0],
        [3, 10, 4, 0],
        [0, 5, 10, 6],
        [0, 0, 7, 10]
    ])
    assert solver_demo.compute_bandwidth(K) == 1


def test_compute_bandwidth_dense_wide():
    K = np.array([
        [10, 0, 0, 1],
        [0, 10, 0, 0],
        [0, 0, 10, 0],
        [2, 0, 0, 10]
    ])
    assert solver_demo.compute_bandwidth(K) == 3


def test_compute_bandwidth_sparse():
    K_dense = np.eye(5)
    K_dense[0, 4] = 1
    K_sparse = sp.csr_matrix(K_dense)
    assert solver_demo.compute_bandwidth(K_sparse) == 4


def test_compute_bandwidth_zero_matrix():
    K = np.zeros((5, 5))
    assert solver_demo.compute_bandwidth(K) == 0  # No nonzeros, so bandwidth is 0
    K_sparse = sp.csr_matrix(K)
    assert solver_demo.compute_bandwidth(K_sparse) == 0


def test_compute_bandwidth_diagonal_only():
    K = np.diag([1, 2, 3, 4])
    assert solver_demo.compute_bandwidth(K) == 0


def test_condition_number_identity_dense():
    K = np.eye(5)
    cond = solver_demo.compute_condition_number(K)
    assert np.isclose(cond, 1.0)


def test_condition_number_scaled_identity_dense():
    K = 10 * np.eye(10)
    cond = solver_demo.compute_condition_number(K)
    assert np.isclose(cond, 1.0)


def test_condition_number_diagonal_ill_conditioned_dense():
    K = np.diag([1, 1e-5, 1e-10])
    cond = solver_demo.compute_condition_number(K)
    assert np.isclose(cond, 1e10, rtol=1e-12)


def test_condition_number_singular_dense():
    K = np.array([
        [1, 2],
        [2, 4]  # rank-deficient
    ])
    cond = solver_demo.compute_condition_number(K)
    assert cond > 1e15


def test_condition_number_sparse_identity():
    K_sparse = sp.eye(1000, format='csr')
    cond = solver_demo.compute_condition_number(K_sparse)
    assert np.isclose(cond, 1.0, rtol=1e-2)  # allow slight estimation error


def test_condition_number_sparse_random_well_conditioned():
    rng = np.random.default_rng(0)
    K_dense = rng.normal(size=(50, 50))
    K_dense = K_dense @ K_dense.T + 50 * np.eye(50)  # make SPD and well-conditioned
    K_sparse = sp.csr_matrix(K_dense)
    cond = solver_demo.compute_condition_number(K_sparse)
    assert cond < 100  # Should be well-conditioned


def test_condition_number_sparse_singular():
    K_sparse = sp.csr_matrix(np.array([
        [1, 2, 0],
        [2, 4, 0],
        [0, 0, 0]
    ]))
    cond = solver_demo.compute_condition_number(K_sparse)
    assert cond > 10e10


def test_sparsity_all_zero_dense():
    K = np.zeros((4, 4))
    assert solver_demo.compute_sparsity(K) == 100.0


def test_sparsity_all_nonzero_dense():
    K = np.ones((5, 5))
    assert solver_demo.compute_sparsity(K) == 0.0


def test_sparsity_half_dense():
    K = np.array([
        [1, 0],
        [0, 1]
    ])
    assert solver_demo.compute_sparsity(K) == 50.0


def test_sparsity_sparse_equivalent():
    K_dense = np.array([
        [1, 0],
        [0, 0]
    ])
    K_sparse = sp.csr_matrix(K_dense)
    assert solver_demo.compute_sparsity(K_sparse) == solver_demo.compute_sparsity(K_dense)


def test_sparsity_large_sparse():
    K_sparse = sp.rand(100, 100, density=0.05, format='csr')
    sparsity = solver_demo.compute_sparsity(K_sparse)
    assert 80.0 < sparsity < 99.0  # allow some noise in random fill


def generate_large_sparse_matrix(n=10):
    """
    Creates a large sparse matrix similar in structure to a 2D FEM stiffness matrix.

    Returns
    -------
    K : scipy.sparse.csr_matrix
        A sparse symmetric positive definite matrix.
    """
    size = n * n  # total number of DOFs
    diagonals = [
        4 * np.ones(size),
        -1 * np.ones(size - 1),
        -1 * np.ones(size - n)
    ]
    offsets = [0, 1, n]
    
    # Start in LIL format for fast editing
    K = sp.diags(diagonals, offsets, shape=(size, size), format='lil')

    # Fix wraparound connections
    for i in range(1, n):
        K[i * n, i * n - 1] = 0
        K[i * n - 1, i * n] = 0

    # Convert to CSR format for efficient numerical use
    return K.tocsr()


def test_analyze_and_visualize_matrix():
    K = generate_large_sparse_matrix()
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    fname = test_dir / "test_matrix_viz_imshow.png"
    solver_demo.analyze_and_visualize_matrix(K, str(fname))
    # Ensure the plot file is created
    assert fname.exists(), f"Plot file {fname} was not created"
    # Ensure the file is non-empty
    assert os.stat(fname).st_size > 0, f"Plot file {fname} was not created"
    fname = test_dir / "test_matrix_viz_spy.png"
    solver_demo.analyze_and_visualize_matrix(K, str(fname), method="spy")
    # Ensure the plot file is created
    assert fname.exists(), f"Plot file {fname} was not created"
    # Ensure the file is non-empty
    assert os.stat(fname).st_size > 0, f"Plot file {fname} was not created"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_visualize_stiffness_matrix(ele_type):
    # create an example system
    x_lower, y_lower = 0, 10
    x_upper, y_upper = 10, 5
    nx, ny = 10, 5
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    fname = test_dir / f"test_mesh_vizK_{ele_type}.png"
    pre_demo.plot_mesh_2D(str(fname), ele_type, coords, connect)
    fname = test_dir / f"test_vizK_{ele_type}.png"
    displacement = np.zeros(coords.shape)
    for kk in range(0, coords.shape[0]):
        displacement[kk, 0] = coords[kk, 0] * 0.1
    method = "imshow"
    material_props = np.asarray([1, 100])
    solver_demo.visualize_stiffness_matrix(fname, ele_type, coords.T, connect.T, material_props, displacement.T, method)
    # Ensure the plot file is created
    assert fname.exists(), f"Plot file {fname} was not created for {ele_type}"
    # Ensure the file is non-empty
    assert os.stat(fname).st_size > 0, f"Plot file {fname} is empty for {ele_type}"


def dummy_sleep(x, delay=0.001):
    time.sleep(delay)
    return x * 2


def test_time_function_call_returns_float():
    avg_time = solver_demo.time_function_call(dummy_sleep, 3, num_runs=3, delay=0.001)
    assert isinstance(avg_time, float)
    assert avg_time > 0


def test_time_function_call_with_positional_args():
    avg_time = solver_demo.time_function_call(dummy_sleep, 3, num_runs=2, delay=0.001)
    assert avg_time > 0
    # Estimate expected time based on delay
    assert 0.0001 < avg_time < 0.05  # loose range to account for variability


def test_time_function_call_with_numpy_function():
    avg_time = solver_demo.time_function_call(np.dot, np.eye(10), np.eye(10), num_runs=5)
    assert avg_time > 0


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_time_assemble_global_stiffness(ele_type):
    num_runs = 2

    # Create a small example mesh
    x_lower, y_lower = 0, 0
    x_upper, y_upper = 10, 10
    nx, ny = 2, 2

    coords, connect = pre.generate_rect_mesh_2d(
        ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny
    )

    # Small dummy displacement field
    displacement = np.zeros(coords.shape)
    for kk in range(coords.shape[0]):
        displacement[kk, 0] = coords[kk, 0] * 0.1

    material_props = np.asarray([1.0, 100.0])

    # Transpose inputs to match expected shape (ncoord, nnode)
    avg_time = solver_demo.time_assemble_global_stiffness(
        num_runs=num_runs,
        ele_type=ele_type,
        coords=coords.T,
        connect=connect.T,
        material_props=material_props,
        displacement=displacement.T
    )

    assert avg_time > 0, f"Assembly time should be > 0 for {ele_type}"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_time_assemble_global_stiffness_sparse(ele_type):
    num_runs = 2

    # Create a small example mesh
    x_lower, y_lower = 0, 0
    x_upper, y_upper = 10, 10
    nx, ny = 2, 2

    coords, connect = pre.generate_rect_mesh_2d(
        ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny
    )

    # Small dummy displacement field
    displacement = np.zeros(coords.shape)
    for kk in range(coords.shape[0]):
        displacement[kk, 0] = coords[kk, 0] * 0.1

    material_props = np.asarray([1.0, 100.0])

    # Transpose inputs to match expected shape (ncoord, nnode)
    avg_time = solver_demo.time_assemble_global_stiffness_sparse(
        num_runs=num_runs,
        ele_type=ele_type,
        coords=coords.T,
        connect=connect.T,
        material_props=material_props,
        displacement=displacement.T
    )

    assert avg_time > 0, f"Assembly time should be > 0 for {ele_type}"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_time_assemble_global_traction(ele_type):
    num_runs = 2

    # Create a small example mesh
    x_lower, y_lower = 0, 0
    x_upper, y_upper = 10, 10
    nx, ny = 2, 2

    coords, connect = pre.generate_rect_mesh_2d(
        ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny
    )

    # Create loading information
    _, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower, x_upper, y_lower, y_upper
    )
    q = 1.0
    dload_info = pre.assign_uniform_load_rect(boundary_edges, "top", 0.0, q)

    avg_time = solver_demo.time_assemble_global_traction(
        num_runs=num_runs,
        ele_type=ele_type,
        coords=coords.T,
        connect=connect.T,
        dload_info=dload_info
    )

    assert avg_time > 0


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_time_assemble_global_residual(ele_type):
    num_runs = 2

    # Create a small example mesh
    x_lower, y_lower = 0, 0
    x_upper, y_upper = 10, 10
    nx, ny = 2, 2

    coords, connect = pre.generate_rect_mesh_2d(
        ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny
    )

    # Small dummy displacement field
    displacement = np.zeros(coords.shape)
    for kk in range(coords.shape[0]):
        displacement[kk, 0] = coords[kk, 0] * 0.1

    material_props = np.asarray([1.0, 100.0])

    # Transpose inputs to match expected shape (ncoord, nnode)
    avg_time = solver_demo.time_assemble_global_residual(
        num_runs=num_runs,
        ele_type=ele_type,
        coords=coords.T,
        connect=connect.T,
        material_props=material_props,
        displacement=displacement.T
    )

    assert avg_time > 0, f"Assembly time should be > 0 for {ele_type}"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_prep_for_matrix_solve(ele_type):
    # Create a small example mesh
    x_lower, y_lower = 0, 0
    x_upper, y_upper = 10, 10
    nx, ny = 2, 2

    coords, connect = pre.generate_rect_mesh_2d(
        ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny
    )

    # Small dummy displacement field
    displacement = np.zeros(coords.shape)
    for kk in range(coords.shape[0]):
        displacement[kk, 0] = coords[kk, 0] * 0.1

    material_props = np.asarray([1.0, 100.0])

    # Identify domain boundaries
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower, x_upper, y_lower, y_upper
    )

    # Create loading information
    q = 1.0
    dload_info = pre.assign_uniform_load_rect(boundary_edges, "top", 0.0, q)

    # Create fixed displacement information
    fixed_nodes = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0.0, 0.0)

    # run the function
    K, R = solver_demo.prep_for_matrix_solve(ele_type, coords.T, connect.T, material_props, displacement.T, fixed_nodes, dload_info)

    # Check behavior
    num_nodes = coords.shape[0]
    num_dofs = num_nodes * 2  #  2D problem
    assert K.shape[0] == num_dofs
    assert K.shape[1] == num_dofs
    assert R.shape[0] == num_dofs


# Fixture to generate a small well-conditioned SPD matrix
@pytest.fixture
def test_system():
    n = 10
    A = np.random.rand(n, n)
    K = A.T @ A + np.eye(n) * 1e-3  # SPD matrix
    R = np.random.rand(n)
    return K, R


@pytest.mark.parametrize("method", ["dense", "sparse", "sparse_iterative"])
def test_time_one_matrix_solve_returns_positive_time(test_system, method):
    K, R = test_system
    avg_time = solver_demo.time_one_matrix_solve(K, R, method=method, num_runs=2)
    assert isinstance(avg_time, float)
    assert avg_time > 0


def test_invalid_method_raises(test_system):
    K, R = test_system
    with pytest.raises(ValueError, match="Unsupported method"):
        solver_demo.time_one_matrix_solve(K, R, method="not_a_method")