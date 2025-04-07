from finiteelementanalysis import assemble_global as ag
from finiteelementanalysis import discretization as di
from finiteelementanalysis import local_element as le
from finiteelementanalysis import pre_process as pre
import numpy as np
import pytest


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"
])
def test_global_stiffness_zero_displacement(ele_type):
    """
    Tests that the global stiffness matrix:
    - has the correct shape
    - is symmetric
    - is nonzero (stiffness exists)
    - is consistent with FEM logic under zero displacement
    """
    # 1. Mesh and discretization setup
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 2, 1)
    nnode = coords.shape[0]

    # 2. Zero displacement
    displacement = np.zeros((ndof, nnode))

    # 3. Material parameters
    material_props = (10.0, 100.0)

    # 4. Compute global stiffness matrix
    K_global = ag.global_stiffness(ele_type, coords.T, connect.T, material_props, displacement)

    # 5. Check shape
    assert K_global.shape == (ndof * nnode, ndof * nnode), f"{ele_type}: incorrect shape: {K_global.shape}"

    # 6. Check symmetry
    assert np.allclose(K_global, K_global.T, atol=1e-12), f"{ele_type}: K_global is not symmetric"

    # 7. Check that stiffness matrix is non-zero
    assert not np.allclose(K_global, 0.0, atol=1e-12), f"{ele_type}: K_global is all zeros"

    # 8. Check that diagonal entries are positive
    diag = np.diag(K_global)
    assert np.all(diag >= 0.0), f"{ele_type}: found negative diagonal entries in K_global"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"
])
def test_global_residual_zero_and_stretched(ele_type):
    """
    Test that the global residual vector:
    - is zero under zero displacement
    - is non-zero under applied deformation
    - has the correct shape
    """
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.5, 0.5, 1.0, 1.0, 2, 1)
    nnode = coords.shape[0]

    material_props = (10.0, 100.0)  # Neo-Hookean: mu, kappa

    # Zero displacement → zero residual
    displacement_zero = np.zeros((ndof, nnode))
    R_zero = ag.global_residual(ele_type, coords.T, connect.T, material_props, displacement_zero)
    assert R_zero.shape == (ndof * nnode,), f"{ele_type}: incorrect residual shape"
    assert np.allclose(R_zero, 0.0, atol=1e-12), f"{ele_type}: residual not zero under zero displacement"

    # Simple stretch → non-zero residual
    def u(x, y): return np.array([0.1 * x, 0.05 * y])
    displacement = np.array([u(x, y) for x, y in coords]).T  # (ndof, nnode)
    R_stretched = ag.global_residual(ele_type, coords.T, connect.T, material_props, displacement)

    assert not np.allclose(R_stretched, 0.0, atol=1e-12), f"{ele_type}: residual unexpectedly zero under deformation"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"
])
def test_linear_fem_consistency_Ku_equals_R(ele_type):
    """
    For small displacements and linear elements, test that:
        R ≈ K @ u
    This validates the linear FEM formulation under small deformation.
    """
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # Create a structured 2x1 mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 2.0, 1.0, 2, 1)
    nnode = coords.shape[0]

    # Define small displacement (quasi-infinitesimal to suppress nonlinearity)
    def u(x, y): return np.array([1e-6 * x, 1e-6 * y])
    displacement = np.array([u(x, y) for x, y in coords]).T  # shape (ndof, nnode)

    # Material properties (linear-ish region of Neo-Hookean)
    mu = 1.0
    kappa = 100.0
    material_props = (mu, kappa)

    # Assemble global stiffness and residual
    K_global = ag.global_stiffness(ele_type, coords.T, connect.T, material_props, displacement)
    R_global = ag.global_residual(ele_type, coords.T, connect.T, material_props, displacement)

    u_flat = displacement.T.flatten()
    R_test = K_global @ u_flat

    # Compare R_test and R_global
    assert np.allclose(R_test, R_global, atol=1e-12), (
        f"{ele_type}: Linear consistency failed\n"
        f"K @ u != R\n"
        f"max diff: {np.max(np.abs(R_test - R_global))}"
    )


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"
])
def test_global_virtual_work_consistency(ele_type):
    """
    Verifies that the internal energy computed by:
      W = u^T K u ≈ u^T r
    is consistent for a small, stretched mesh.
    """
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # Mesh: 2x1 element block
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 2, 1)
    nnode = coords.shape[0]

    # Displacement field: uniform stretch
    def u(x, y): return np.array([0.0001 * x, 0.00005 * y])
    displacement = np.array([u(x, y) for x, y in coords]).T  # shape: (ndof, nnode)

    material_props = (10.0, 100.0)  # mu, kappa

    # Compute global stiffness and residual
    K_global = ag.global_stiffness(ele_type, coords.T, connect.T, material_props, displacement)
    R_global = ag.global_residual(ele_type, coords.T, connect.T, material_props, displacement)

    # Flatten displacement
    u_flat = displacement.T.flatten()

    # Compute virtual work two ways
    W_K = u_flat @ K_global @ u_flat
    W_R = u_flat @ R_global

    # Assert they are consistent
    assert np.isclose(W_K, W_R, atol=1e-9), (
        f"{ele_type}: Virtual work mismatch\n"
        f"u^T K u = {W_K:.8e}, u^T r = {W_R:.8e}"
    )


@pytest.mark.parametrize("ele_type, face_id", [
    ("D2_nn3_tri", 0),
    ("D2_nn4_quad", 0),
])
def test_global_traction_single_element_single_face(ele_type, face_id):
    """
    Test that global_traction produces the correct result for a single flat face
    of known length = 1.0 when a uniform traction is applied.
    """
    ncoord, ndof, _ = di.element_info(ele_type)

    # === Build a simple 1-element mesh over [0,1] x [0,1] ===
    x_lower = 0
    y_lower = 0
    x_upper = 1
    y_upper = 1
    nx = 5
    ny = 5
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    coords = coords.T
    connect = connect.T
    nnode = coords.shape[1]

    # === Manually build dload_info for a single face ===
    traction = np.array([2.0, 0.0])  # applied in x
    # dload_info = np.zeros((ndof + 2, 1))
    # dload_info[0, 0] = 0          # element ID
    # dload_info[1, 0] = face_id    # face ID
    # dload_info[2:, 0] = traction  # traction components
    coords_nodewise = coords.T  # shape (nnode, 2)
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords_nodewise, connect.T, ele_type,
        x_lower=x_lower, x_upper=x_upper,
        y_lower=y_lower, y_upper=y_upper
    )
    dload_info = pre.assign_uniform_load_rect(boundary_edges, boundary="right",
                                             dof_0_load=traction[0], dof_1_load=traction[1])

    # === Compute global traction vector ===
    F_global = ag.global_traction(ele_type, coords, connect, dload_info)

    # === Check shape ===
    assert F_global.shape == (ndof * nnode,), f"{ele_type}: incorrect global force shape"

    # === Sum total force from global vector ===
    total_force = np.zeros(ndof)
    for a in range(nnode):
        for i in range(ndof):
            total_force[i] += F_global[ndof * a + i]

    # === Expect total force = traction × 1.0 (length of edge)
    expected_force = traction * 1.0
    np.testing.assert_allclose(total_force, expected_force, rtol=1e-12,
        err_msg=f"{ele_type}: total force mismatch. Expected {expected_force}, got {total_force}")


@pytest.mark.parametrize("ele_type, face_id", [
    ("D2_nn6_tri", 0),
    ("D2_nn8_quad", 0),
])
def test_global_traction_single_face_quadratic(ele_type, face_id):
    """
    For a single quadratic element and a single curved face,
    test that global_traction computes the correct total applied force
    by comparing with numerically integrated expected force using the same quadrature rule.
    """
    ncoord, ndof, _ = di.element_info(ele_type)

    # === Generate a square mesh ===
    x_lower = 0
    y_lower = 0
    x_upper = 1
    y_upper = 1
    nx = 5
    ny = 5
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    coords = coords.T  # shape (ncoord, nnode)
    connect = connect.T  # shape (nelnodes, nelem)
    nnode = coords.shape[1]

    # === Apply uniform traction on one face ===
    traction = np.array([2.0, 0.0])  # x-direction only

    # Build dload_info for just one face
    coords_nodewise = coords.T  # shape (nnode, 2)
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords_nodewise, connect.T, ele_type,
        x_lower=x_lower, x_upper=x_upper,
        y_lower=y_lower, y_upper=y_upper
    )

    dload_info = pre.assign_uniform_load_rect(boundary_edges, boundary="right",
                                             dof_0_load=traction[0], dof_1_load=traction[1])

    # === Compute global traction vector ===
    F_global = ag.global_traction(ele_type, coords, connect, dload_info)

    # === Sum total force from global vector ===
    total_force = np.zeros(ndof)
    for a in range(nnode):
        for i in range(ndof):
            total_force[i] += F_global[ndof * a + i]

    # === Expect total force = traction × 1.0 (length of edge)
    expected_force = traction * 1.0
    np.testing.assert_allclose(total_force, expected_force, rtol=1e-12,
        err_msg=f"{ele_type}: total force mismatch. Expected {expected_force}, got {total_force}")



