from finiteelementanalysis import solver as solver
from finiteelementanalysis import pre_process as pre
import numpy as np
import pytest

element_types = ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"]


@pytest.mark.parametrize("ele_type", element_types)
def test_hyperelastic_solver_basic(ele_type):
    """
    Basic test for the hyperelastic_solver on a small problem to ensure convergence,
    correct shape of output, and no runtime errors.
    """
    ndof = 2

    # 2x2 element mesh of unit square
    x_lower = 0
    y_lower = 0
    x_upper = 10
    y_upper = 5
    nx = 4
    ny = 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower, x_upper, y_lower, y_upper)
    fixed_nodes = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0, 0)
    dload_info = pre.assign_uniform_load_rect(boundary_edges, "right", 0.1, 0.0)

    material_props = np.array([1.0, 10.0])  # Neo-Hookean: [mu, K]

    # Run solver
    displacements_all, nr_print_info_all = solver.hyperelastic_solver(
        material_props,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,
        nr_num_steps=2,
        nr_tol=1e-8,
        nr_maxit=20,
    )

    # Basic assertions
    assert isinstance(displacements_all, list)
    assert len(displacements_all) == 2
    for disp in displacements_all:
        assert isinstance(disp, np.ndarray)
        assert disp.shape[0] == (coords.shape[0] * ndof)
    assert len(nr_print_info_all) == 2


@pytest.mark.parametrize("ele_type", element_types)
def test_zero_load_response(ele_type):
    """
    If no external load is applied (zero traction), and there are no prescribed displacements 
    except the trivial zero solution, the computed displacement should remain zero.
    """
    ndof = 2

    # Create a small mesh (e.g., a unit square)
    x_lower, y_lower, x_upper, y_upper = 0, 0, 1, 1
    nx, ny = 2, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    # No load: assign an empty distributed load
    dload_info = np.empty((ndof + 2, 0))  # no distributed load
    # No prescribed displacement (or fix nothing)
    fixed_nodes = np.empty((3, 0))
    material_props = np.array([1.0, 10.0])

    displacements_all, _ = solver.hyperelastic_solver(
        material_props,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,
        nr_num_steps=1,
        nr_tol=1e-10,
        nr_maxit=10,
    )
    # Expect zero displacement vector
    final_disp = displacements_all[-1]
    np.testing.assert_allclose(final_disp, 0.0, atol=1e-10)


@pytest.mark.parametrize("ele_type", element_types)
def test_prescribed_boundary_conditions(ele_type):
    """
    For a simple case where the boundary displacements are prescribed, check that
    the final solution exactly matches those values at the fixed DOFs.
    """
    ndof = 2

    # Use a small mesh
    x_lower, y_lower, x_upper, y_upper = 0, 0, 1, 1
    nx, ny = 2, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    # Identify boundaries and prescribe a nonzero displacement on the left boundary
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower, x_upper, y_lower, y_upper
    )
    # For example, set x-displacement = 0.1 at the left boundary and y = 0
    fixed_nodes = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0, 0.1)
    # No external load
    dload_info = np.empty((ndof + 2, 0))
    material_props = np.array([1.0, 10.0])

    displacements_all, _ = solver.hyperelastic_solver(
        material_props,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,
        nr_num_steps=1,
        nr_tol=1e-10,
        nr_maxit=20,
    )
    final_disp = displacements_all[-1]

    # Check that for every fixed DOF, the displacement is as prescribed
    num_fixed = fixed_nodes.shape[1]
    for n in range(num_fixed):
        node_id = fixed_nodes[0, n]
        dof_id = int(fixed_nodes[1, n])
        bc_dof = int(ndof * node_id + dof_id)
        assert np.isclose(final_disp[bc_dof], fixed_nodes[2, n])


@pytest.mark.parametrize("ele_type", element_types)
def test_monotonic_loading(ele_type):
    """
    Verify that as the load factor increases (more load steps), the norm of the displacement 
    vector does not decrease.
    """
    x_lower, y_lower, x_upper, y_upper = 0, 0, 10, 5
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower, x_upper, y_lower, y_upper
    )
    fixed_nodes = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0, 0)
    dload_info = pre.assign_uniform_load_rect(boundary_edges, "right", 2.0, 0.0)
    material_props = np.array([1.0, 10.0])

    displacements_all, _ = solver.hyperelastic_solver(
        material_props,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,
        nr_num_steps=5,
        nr_tol=1e-8,
        nr_maxit=20,
    )

    # Compute norms of displacement vectors and check they are monotonic (non-decreasing)
    norms = [np.linalg.norm(disp) for disp in displacements_all]
    # Allow for small numerical oscillations; check that later load step norm is 
    # not significantly less than the previous step.
    for i in range(1, len(norms)):
        assert norms[i] + 1e-12 >= norms[i-1], f"Displacement norm decreased from step {i-1} to {i}"


@pytest.mark.parametrize("ele_type", element_types)
def test_material_parameter_sensitivity(ele_type):
    """
    Verify that making the material stiffer (increasing the bulk modulus)
    results in a smaller displacement for the same applied load.
    """
    ndof = 2

    x_lower, y_lower, x_upper, y_upper = 0, 0, 10, 5
    nx, ny = 4, 2
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower, x_upper, y_lower, y_upper
    )
    fixed_nodes = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0, 0)
    dload_info = pre.assign_uniform_load_rect(boundary_edges, "right", 2.0, 0.0)

    # Material properties: Case 1 (softer) and Case 2 (stiffer)
    material_props_soft = np.array([1.0, 10.0])
    material_props_stiff = np.array([1.0, 100.0])

    disp_soft_all, _ = solver.hyperelastic_solver(
        material_props_soft,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,
        nr_num_steps=3,
        nr_tol=1e-8,
        nr_maxit=30,
    )
    disp_stiff_all, _ = solver.hyperelastic_solver(
        material_props_stiff,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,
        nr_num_steps=3,
        nr_tol=1e-8,
        nr_maxit=30,
    )
    disp_soft = disp_soft_all[-1]
    disp_stiff = disp_stiff_all[-1]
    norm_soft = np.linalg.norm(disp_soft)
    norm_stiff = np.linalg.norm(disp_stiff)
    assert norm_stiff < norm_soft, f"Expected stiffer material to yield lower displacements (norm {norm_stiff} vs {norm_soft})"

