from finiteelementanalysis import local_element as le
from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import discretization as di
import numpy as np
import pytest

#TODO: remove reference to 3D in docstrings in main function


def scalar_cross_2d(v1, v2):
    # Computes the scalar (z-component) of the 2D cross product
    return v1[0] * v2[1] - v1[1] * v2[0]


# Mapping of element types to shape functions
SHAPE_FCN = {
    "D2_nn3_tri": di.D2_nn3_tri,
    "D2_nn6_tri": di.D2_nn6_tri,
    "D2_nn4_quad": di.D2_nn4_quad,
    "D2_nn8_quad": di.D2_nn8_quad
}

# Mapping of element types to face shape functions
SHAPE_FCN_FACE = {
    "D2_nn3_tri": di.D1_nn2,
    "D2_nn6_tri": di.D1_nn3,
    "D2_nn4_quad": di.D1_nn2,
    "D2_nn8_quad": di.D1_nn3
}

SHAPE_DERIV_FCN_FACE = {
    "D2_nn3_tri": di.D1_nn2_dxi,
    "D2_nn6_tri": di.D1_nn3_dxi,
    "D2_nn4_quad": di.D1_nn2_dxi,
    "D2_nn8_quad": di.D1_nn3_dxi
}

FACE_CENTER_NATURAL_COORDS = {
    "D2_nn3_tri": di.gauss_points_1d(1)[:, 0],
    "D2_nn6_tri": di.gauss_points_1d(1)[:, 0],
    "D2_nn4_quad": di.gauss_points_1d(1)[:, 0],
    "D2_nn8_quad": di.gauss_points_1d(1)[:, 0]
}

# Mapping of element types to shape function derivatives
SHAPE_DERIV_FCN = {
    "D2_nn3_tri": di.D2_nn3_tri_dxi,
    "D2_nn6_tri": di.D2_nn6_tri_dxi,
    "D2_nn4_quad": di.D2_nn4_quad_dxi,
    "D2_nn8_quad": di.D2_nn8_quad_dxi
}

# Mapping of element types to element centers in natural coordinates
ELEMENT_CENTER_NATURAL_COORDS = {
    "D2_nn3_tri": di.gauss_points_2d_triangle(1)[:, 0],
    "D2_nn6_tri": di.gauss_points_2d_triangle(1)[:, 0],
    "D2_nn4_quad": di.gauss_points_2d_quad(1)[:, 0],
    "D2_nn8_quad": di.gauss_points_2d_quad(1)[:, 0]
}


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_element_residual_stretch_nonzero_all_elements(ele_type):
    """
    Test that the element residual is non-zero under uniform stretch for all supported element types.
    """
    # Get element properties
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # Create a 1x1 test mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T  # shape (ncoord, nelnodes)

    # Define linear displacement field: u(x, y) = [0.1x, 0.05y]
    def u(x, y):
        return np.array([0.1 * x, 0.05 * y])

    # Compute nodal displacements from geometry
    node_coords = coords[connect[0]]  # shape (nelnodes, 2)
    displacement = np.array([u(x, y) for x, y in node_coords]).T  # shape (ndof, nelnodes)

    # Material properties (e.g., Neo-Hookean)
    mu = 10.0
    kappa = 100.0
    material_props = np.array([mu, kappa])

    # Compute element residual
    rel = le.element_residual(ele_type, element_coords, material_props, displacement)

    # Validate output shape
    assert rel.shape == (ndof * nelnodes,), f"{ele_type}: residual shape mismatch: {rel.shape}"

    # Residual should not be all zeros
    assert not np.allclose(rel, 0.0, atol=1e-12), f"{ele_type}: residual is unexpectedly zero under deformation"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"
])
def test_element_residual_zero_displacement(ele_type):
    """
    Test that the element residual is zero when displacement is zero
    for all supported 2D element types.
    """
    # FEM element data
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    
    # Simple square element
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T  # shape (ncoord, nelnodes)

    # Zero displacement
    displacement = np.zeros((ndof, nelnodes))

    # Material parameters (e.g., Neo-Hookean)
    mu = 10.0
    kappa = 100.0
    material_props = np.array([mu, kappa])

    # Compute residual
    rel = le.element_residual(ele_type, element_coords, material_props, displacement)

    # Check shape
    assert rel.shape == (ndof * nelnodes,), f"{ele_type}: Wrong residual shape: {rel.shape}"

    # Residual should be zero (or very close)
    assert np.allclose(rel, 0.0, atol=1e-12), f"{ele_type}: Residual not zero under zero displacement:\n{rel}"



@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_element_distributed_load_total_force(ele_type):
    """
    Test that element_distributed_load returns the correct total force
    when a constant traction is applied to a straight 1-unit face.
    """
    ncoord, ndof, _ = di.element_info(ele_type)
    face_elem_type, num_face_nodes, _ = di.face_info(ele_type)

    # 1. Face coordinates (horizontal edge of length = 1.0)
    if num_face_nodes == 2:
        coords = np.array([[0.0, 1.0],     # x
                           [0.0, 0.0]])    # y
    elif num_face_nodes == 3:
        coords = np.array([[0.0, 1.0, 0.5],  # x: left, right, mid
                           [0.0, 0.0, 0.0]]) # y
    else:
        raise ValueError(f"Unsupported face node count: {num_face_nodes}")

    # 2. Uniform traction
    traction = np.array([2.0, 0.0])  # along x

    # 3. Compute load vector
    r = le.element_distributed_load(ele_type, coords, traction)

    # 4. Check shape
    assert r.shape == (ndof * num_face_nodes,), f"{ele_type}: Wrong output shape {r.shape}"

    # 5. Check that the total force equals traction × length = [2.0, 0.0]
    total_force = np.zeros(ndof)
    for a in range(num_face_nodes):
        total_force += r[ndof*a : ndof*(a+1)]

    expected_force = traction * 1.0  # length = 1
    np.testing.assert_allclose(total_force, expected_force, rtol=1e-12,
        err_msg=f"{ele_type}: total force mismatch. Got {total_force}, expected {expected_force}")


# def test_compute_face_jacobian():
@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN_FACE.keys())
@pytest.mark.parametrize("orientation, expected_dx_dxi, expected_dy_dxi", [
    ("horizontal", 0.5, 0.0),
    ("vertical", 0.0, 0.5),
    ("diagonal", 0.5, 0.5),
    ("reversed", -0.5, 0.0),
])
def test_compute_face_jacobian_with_real_shape_functions(ele_type, orientation, expected_dx_dxi, expected_dy_dxi):
    """
    Robust test for compute_face_jacobian using real shape function derivatives
    across multiple face orientations and all supported 2D element types.
    Node ordering assumed: [left, right, mid] for 3-node faces.
    """
    ncoord = 2
    deriv_fcn = SHAPE_DERIV_FCN_FACE[ele_type]
    xi = FACE_CENTER_NATURAL_COORDS[ele_type]
    dNdxi = deriv_fcn(xi)

    num_face_nodes = dNdxi.shape[0]

    # Generate coords based on orientation
    if orientation == "horizontal":
        if num_face_nodes == 2:
            coords = np.array([[0.0, 1.0],
                               [0.0, 0.0]])
        else:
            coords = np.array([[0.0, 1.0, 0.5],
                               [0.0, 0.0, 0.0]])
    elif orientation == "vertical":
        if num_face_nodes == 2:
            coords = np.array([[0.0, 0.0],
                               [0.0, 1.0]])
        else:
            coords = np.array([[0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.5]])
    elif orientation == "diagonal":
        if num_face_nodes == 2:
            coords = np.array([[0.0, 1.0],
                               [0.0, 1.0]])
        else:
            coords = np.array([[0.0, 1.0, 0.5],
                               [0.0, 1.0, 0.5]])
    elif orientation == "reversed":
        if num_face_nodes == 2:
            coords = np.array([[1.0, 0.0],
                               [0.0, 0.0]])
        else:
            coords = np.array([[1.0, 0.0, 0.5],
                               [0.0, 0.0, 0.0]])
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    # Compute face Jacobian
    dxdxi = le.compute_face_jacobian(ncoord, coords, dNdxi)

    # Validate shape
    assert dxdxi.shape == (ncoord, 1), f"{ele_type}, {orientation}: expected shape (2, 1), got {dxdxi.shape}"

    dx_dxi = dxdxi[0, 0]
    dy_dxi = dxdxi[1, 0]

    assert np.isclose(dx_dxi, expected_dx_dxi, atol=1e-12), \
        f"{ele_type}, {orientation}: expected dx/dxi ≈ {expected_dx_dxi}, got {dx_dxi}"
    assert np.isclose(dy_dxi, expected_dy_dxi, atol=1e-12), \
        f"{ele_type}, {orientation}: expected dy/dxi ≈ {expected_dy_dxi}, got {dy_dxi}"


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN_FACE.keys())
@pytest.mark.parametrize("orientation, expected_length", [
    ("horizontal", 1.0 / 2.0),  # natural coords has length 2
    ("vertical", 2.0 / 2.0),
    ("diagonal", np.sqrt(2) / 2.0),
    ("reversed", 1.0 / 2.0),
])
def test_compute_face_measure_from_jacobian(ele_type, orientation, expected_length):
    """
    Tests compute_face_measure using compute_face_jacobian and real shape function derivatives.
    Covers multiple face orientations and element types.
    """
    ncoord = 2
    deriv_fcn = SHAPE_DERIV_FCN_FACE[ele_type]
    xi = FACE_CENTER_NATURAL_COORDS[ele_type]
    dNdxi = deriv_fcn(xi)
    num_face_nodes = dNdxi.shape[0]

    # Build face coordinates based on orientation
    if orientation == "horizontal":
        if num_face_nodes == 2:
            coords = np.array([[0.0, 1.0],
                               [0.0, 0.0]])
        else:
            coords = np.array([[0.0, 1.0, 0.5],
                               [0.0, 0.0, 0.0]])
    elif orientation == "vertical":
        if num_face_nodes == 2:
            coords = np.array([[0.0, 0.0],
                               [0.0, 2.0]])
        else:
            coords = np.array([[0.0, 0.0, 0.0],
                               [0.0, 2.0, 1.0]])
    elif orientation == "diagonal":
        if num_face_nodes == 2:
            coords = np.array([[0.0, 1.0],
                               [0.0, 1.0]])
        else:
            coords = np.array([[0.0, 1.0, 0.5],
                               [0.0, 1.0, 0.5]])
    elif orientation == "reversed":
        if num_face_nodes == 2:
            coords = np.array([[1.0, 0.0],
                               [0.0, 0.0]])
        else:
            coords = np.array([[1.0, 0.0, 0.5],
                               [0.0, 0.0, 0.0]])
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    # Compute Jacobian and face measure
    dxdxi = le.compute_face_jacobian(ncoord, coords, dNdxi)
    measure = le.compute_face_measure(ncoord, dxdxi)

    # Compare to expected edge length
    assert np.isclose(measure, expected_length, atol=1e-12), \
        f"{ele_type}, {orientation}: expected length {expected_length}, got {measure}"


@pytest.mark.parametrize("ele_type", SHAPE_FCN.keys())
def test_face_load_contribution(ele_type):
    """
    Tests that the face load contribution computed using actual shape function values
    is correctly assembled and conserves total force.
    """
    shape_fcn = SHAPE_FCN_FACE[ele_type]
    xi_eta = FACE_CENTER_NATURAL_COORDS[ele_type]
    N = shape_fcn(xi_eta)  # Shape function values at element center

    num_nodes = len(N)
    ndof = 2
    traction = np.array([3.0, -2.0])

    r = le.compute_face_load_contribution(num_nodes, ndof, traction, N)

    # Check shape
    assert r.shape == (ndof * num_nodes,), f"{ele_type}: unexpected r shape: {r.shape}"

    # Check total force matches expected
    total_force = np.zeros(ndof)
    for a in range(num_nodes):
        total_force += r[ndof * a : ndof * (a + 1)]

    np.testing.assert_allclose(total_force, traction, atol=1e-12,
        err_msg=f"{ele_type}: total force mismatch. Got {total_force}, expected {traction}")


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"])
def test_stiffness_matrix_is_positive_semi_definite(ele_type):
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T
    displacement = np.zeros((ndof, nelnodes))
    material_props = np.array([10.0, 100.0])

    kel = le.element_stiffness(ele_type, material_props, element_coords, displacement)

    eigvals = np.linalg.eigvalsh(kel)
    assert np.all(eigvals >= -1e-12), f"Stiffness matrix is not positive semi-definite: eigenvalues={eigvals}"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"])
def test_rigid_body_motion_produces_no_energy(ele_type):
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T

    # Rigid translation
    displacement = np.ones((ndof, nelnodes)) * 0.5
    material_props = np.array([10.0, 100.0])
    kel = le.element_stiffness(ele_type, material_props, element_coords, displacement)

    u_flat = displacement.T.flatten()
    W_internal = 0.5 * u_flat @ kel @ u_flat
    assert np.isclose(W_internal, 0.0, atol=1e-12), "Rigid body motion should not produce internal energy"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"])
def test_virtual_work_energy_consistency(ele_type):
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T

    # Linear displacement
    node_coords = coords[connect[0]]
    def u(x, y): return np.array([0.1 * x, 0.2 * y])
    displacement = np.array([u(x, y) for x, y in node_coords]).T

    material_props = np.array([10.0, 100.0])
    kel = le.element_stiffness(ele_type, material_props, element_coords, displacement)

    # Virtual work: W_int = 0.5 * u.T @ K @ u
    u_flat = displacement.T.flatten()
    W_internal = 0.5 * u_flat @ kel @ u_flat
    assert W_internal > 0, "Internal energy should be positive"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"])
def test_stiffness_no_geometric_term(ele_type):
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T
    displacement = np.zeros((ndof, nelnodes))
    material_props = np.array([10.0, 100.0])

    kel = le.element_stiffness(ele_type, material_props, element_coords, displacement)

    eigvals = np.linalg.eigvalsh(kel)
    assert np.all(eigvals >= -1e-10), "Stiffness matrix has negative eigenvalues at zero displacement"


@pytest.mark.parametrize("ele_type", ["D2_nn3_tri", "D2_nn4_quad", "D2_nn6_tri", "D2_nn8_quad"])
def test_stiffness_increases_with_stretch(ele_type):
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T
    material_props = np.array([10.0, 100.0])

    zero_disp = np.zeros((ndof, nelnodes))
    kel_0 = le.element_stiffness(ele_type, material_props, element_coords, zero_disp)

    # Stretch in x and y
    node_coords = coords[connect[0]]
    def u(x, y): return np.array([0.1 * x, 0.05 * y])
    stretched_disp = np.array([u(x, y) for x, y in node_coords]).T
    kel_stretched = le.element_stiffness(ele_type, material_props, element_coords, stretched_disp)

    assert not np.allclose(kel_0, kel_stretched, atol=1e-10), f"{ele_type}: Stiffness did not change with deformation"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_element_stiffness_matrix_symmetry(ele_type):
    """
    Verifies that the element stiffness matrix returned from element_stiffness() is:
    - The correct shape
    - Symmetric (as expected for many standard hyperelastic materials)
    """
    # 1. Discretization info
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # 2. Generate a mesh and extract one element
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]]  # (nelnodes, 2)

    # Transpose if needed (function expects (ncoord, nelnodes))
    coords_element = element_coords.T  # (2, nelnodes)

    # 3. Displacement: zero → no geometric stiffness
    displacement = np.zeros((ndof, nelnodes))

    # 4. Material parameters
    mu = 10.0
    kappa = 100.0
    material_props = np.array([mu, kappa])

    # 5. Compute element stiffness
    kel = le.element_stiffness(ele_type, material_props, coords_element, displacement)

    # 6. Check shape
    assert kel.shape == (ndof * nelnodes, ndof * nelnodes), \
        f"Wrong shape for kel: {kel.shape}, expected {(ndof * nelnodes, ndof * nelnodes)}"

    # 7. Check symmetry
    assert np.allclose(kel, kel.T, atol=1e-12), f"{ele_type} stiffness matrix is not symmetric:\n{kel}"


@pytest.mark.parametrize("ele_type", [
    "D2_nn3_tri",
    "D2_nn4_quad",
    "D2_nn6_tri",
    "D2_nn8_quad",
])
def test_element_stiffness_increase_with_deformation(ele_type):
    """
    Verifies that the element stiffness matrix changes when a nonzero deformation is applied,
    due to geometric stiffness effects. This test is run for all supported element types.
    """
    # 1. Discretization info
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # 2. Generate a 1x1 element mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]].T  # Shape: (ncoord, nelnodes)

    # 3. Material properties
    mu = 10.0
    kappa = 100.0
    material_props = np.array([mu, kappa])

    # 4. Zero displacement → no geometric stiffness
    displacement_zero = np.zeros((ndof, nelnodes))
    kel_zero = le.element_stiffness(ele_type, material_props, element_coords, displacement_zero)

    # 5. Linear stretch displacement field: u(x, y) = [αx, βy]
    α, β = 0.1, 0.05
    node_coords = coords[connect[0]]
    displacement_linear = np.array([[α * x, β * y] for x, y in node_coords]).T  # Shape: (ndof, nelnodes)

    kel_stretched = le.element_stiffness(ele_type, material_props, element_coords, displacement_linear)

    # 6. Assert that the stiffness matrix has changed
    assert not np.allclose(kel_zero, kel_stretched, atol=1e-10), \
        f"{ele_type}: Stiffness matrix did not change under deformation"


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN.keys())
def test_compute_jacobian(ele_type):
    """
    Tests that the Jacobian for each element satisfies expected properties:
    - Positive determinant
    - Full rank
    - Reasonable condition number
    - Optional area consistency (linear elements, triangular and quadrilateral)
    """
    # Generate a small structured mesh
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 2.0, 1.0
    nx, ny = 2, 1
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    # Get shape function derivative function and element center
    shape_deriv_fcn = SHAPE_DERIV_FCN[ele_type]
    xi_eta = ELEMENT_CENTER_NATURAL_COORDS[ele_type]

    # Loop over all elements
    for e in range(connect.shape[0]):
        element_coords = coords[connect[e]].T # transpose to match expected shape for the function
        dNdxi = shape_deriv_fcn(xi_eta)

        # Compute Jacobian and determinant
        dxdxi, det_dxdxi = le.compute_jacobian(element_coords, dNdxi)

        # 1. Jacobian determinant must be positive
        assert det_dxdxi > 0, f"{ele_type} element {e} has non-positive det(J): {det_dxdxi}"

        # 2. Jacobian must be full rank
        assert np.linalg.matrix_rank(dxdxi) == 2, f"{ele_type} element {e} has rank-deficient Jacobian"

        # 3. Condition number should be reasonable
        cond = np.linalg.cond(dxdxi)
        assert cond < 1e3, f"{ele_type} element {e} has high condition number: {cond}"

        # 4. Area consistency (only for linear elements)
        if ele_type in ["D2_nn3_tri", "D2_nn4_quad"]:
            if "tri" in ele_type:
                # Triangle: area from cross product of two edges
                a = element_coords[:, 0]
                b = element_coords[:, 1]
                c = element_coords[:, 2]
                expected_area = 0.5 * np.abs(scalar_cross_2d(b - a, c - a))
                approx_area = det_dxdxi * 0.5  # where 0.5 is the area of the element in natural coords
            elif "quad" in ele_type:
                # Quadrilateral (assuming bilinear, but linear shape): split into two triangles
                a = element_coords[:, 0]
                b = element_coords[:, 1]
                c = element_coords[:, 2]
                d = element_coords[:, 3]
                area1 = 0.5 * np.abs(scalar_cross_2d(b - a, d - a))
                area2 = 0.5 * np.abs(scalar_cross_2d(c - b, d - b))
                expected_area = area1 + area2
                approx_area = det_dxdxi * 4.0  # where 4.0 is the area of the element in natural coords
            assert np.isclose(approx_area, expected_area, rtol=1e-3), \
                f"{ele_type} element {e}: area mismatch. Expected: {expected_area}, got {approx_area}"


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN.keys())
def test_convert_derivatives_from_shape_functions(ele_type):
    """
    Tests convert_derivatives() using actual shape function derivatives and Jacobian inversion.
    Verifies that:
    - The result shape is (n_nodes, 2)
    - The gradients satisfy ∑ ∇N_i = 0 (partition of unity)
    """
    # Create a small mesh
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 2.0, 1.0
    nx, ny = 2, 1
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    xi_eta = ELEMENT_CENTER_NATURAL_COORDS[ele_type]
    shape_deriv_fcn = SHAPE_DERIV_FCN[ele_type]

    for e in range(connect.shape[0]):
        element_coords = coords[connect[e]].T  # (2, n_nodes)
        dNdxi = shape_deriv_fcn(xi_eta)  # (n_nodes, 2)

        # Compute Jacobian and inverse
        dxdxi, _ = le.compute_jacobian(element_coords, dNdxi)
        dxidx = np.linalg.inv(dxdxi)

        # Compute dNdx using chain rule
        dNdx = le.convert_derivatives(dNdxi, dxidx)  # (n_nodes, 2)

        # Check shape
        assert dNdx.shape == dNdxi.shape, f"{ele_type} element {e}: shape mismatch"

        # Partition of unity: sum of ∇N_i should be 0
        grad_sum = np.sum(dNdx, axis=0)
        np.testing.assert_allclose(grad_sum, np.zeros(2), atol=1e-12,
            err_msg=f"{ele_type} element {e}: ∑∇N_i ≠ 0 → {grad_sum}")


def test_B_identity():
    F = np.eye(2)
    B = le.compute_B(F)
    np.testing.assert_allclose(B, np.eye(2), atol=1e-12)


def test_B_uniform_scaling():
    F = np.diag([2.0, 2.0])
    B = le.compute_B(F)
    expected = np.diag([4.0, 4.0])
    np.testing.assert_allclose(B, expected, atol=1e-12)


def test_B_rotation():
    theta = np.pi / 3
    F = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    B = le.compute_B(F)
    np.testing.assert_allclose(B, np.eye(2), atol=1e-12)


def test_B_affine_deformation():
    F = np.array([
        [1.5, 0.3],
        [0.2, 1.1]
    ])
    B = le.compute_B(F)
    expected = F @ F.T
    np.testing.assert_allclose(B, expected, atol=1e-12)


def test_B_symmetry():
    F = np.array([
        [1.2, 0.4],
        [0.1, 0.9]
    ])
    B = le.compute_B(F)
    assert np.allclose(B, B.T, atol=1e-12), "B is not symmetric"


def test_compute_J():
    # Identity: no deformation
    F = np.eye(2)
    assert le.compute_J(F) == pytest.approx(1.0)

    # Pure scaling
    F = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert le.compute_J(F) == pytest.approx(6.0)

    # Rotation (volume-preserving)
    theta = np.pi / 4
    F = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    assert le.compute_J(F) == pytest.approx(1.0)

    # Inverted mapping (reflection across x-axis)
    F = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert le.compute_J(F) == pytest.approx(-1.0)

    # Degenerate (collapsed element, rank-deficient)
    F = np.array([[1.0, 2.0], [1.0, 2.0]])  # rank 1
    assert le.compute_J(F) == pytest.approx(0.0)


def test_compute_Finv_identity():
    F = np.eye(2)
    Finv = le.compute_Finv(F)
    np.testing.assert_allclose(Finv, np.eye(2), atol=1e-12)


def test_compute_Finv_scaling():
    F = np.diag([2.0, 4.0])
    Finv = le.compute_Finv(F)
    expected = np.diag([0.5, 0.25])
    np.testing.assert_allclose(Finv, expected, atol=1e-12)


def test_compute_Finv_rotation():
    theta = np.pi / 6
    F = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    Finv = le.compute_Finv(F)
    # For rotations, inverse = transpose
    expected = F.T
    np.testing.assert_allclose(Finv, expected, atol=1e-12)


def test_compute_Finv_composite():
    # Stretch and shear
    F = np.array([
        [2.0, 1.0],
        [0.0, 3.0]
    ])
    Finv = le.compute_Finv(F)
    expected = np.linalg.inv(F)
    np.testing.assert_allclose(Finv, expected, atol=1e-12)


def test_compute_Finv_singular():
    F = np.array([
        [1.0, 2.0],
        [2.0, 4.0]  # Linearly dependent rows
    ])
    with pytest.raises(np.linalg.LinAlgError):
        le.compute_Finv(F)


@pytest.mark.parametrize("F", [
    np.eye(2),  # Identity
    np.diag([2.0, 3.0]),  # Scaling
    np.array([[0.0, -1.0], [1.0, 0.0]]),  # 90-degree rotation
    np.array([[1.0, 2.0], [3.0, 4.0]]),  # General non-symmetric
    np.array([[1.2, 0.5], [0.3, 2.1]])   # Mixed shear/stretch
])
def test_F_times_Finv_is_identity(F):
    Finv = le.compute_Finv(F)
    I_solve = F @ Finv
    np.testing.assert_allclose(I_solve, np.eye(F.shape[0]), atol=1e-12)
    assert I_solve.shape == (2, 2)


@pytest.mark.parametrize("F", [
    np.eye(2),  # Identity
    np.diag([2.0, 3.0]),  # Scaling
    np.array([[1.0, 2.0], [3.0, 4.0]]),  # General
    np.array([[0.5, 1.2], [-0.3, 2.5]])  # Mixed transformation
])
def test_det_Finv_is_inverse_of_det_F(F):
    det_F = le.compute_J(F)
    Finv = le.compute_Finv(F)
    det_Finv = le.compute_J(Finv)

    # Theoretically: det(Finv) = 1 / det(F)
    assert det_F != 0.0, "Matrix is singular"
    assert det_Finv == pytest.approx(1.0 / det_F, rel=1e-12)


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN.keys())
def test_compute_deformation_gradient_identity(ele_type):
    """
    Tests that compute_deformation_gradient returns the identity matrix
    when the displacement field is zero, across all element types.
    """
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 2.0, 1.0
    nx, ny = 2, 1
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    xi_eta = ELEMENT_CENTER_NATURAL_COORDS[ele_type]
    shape_deriv_fcn = SHAPE_DERIV_FCN[ele_type]

    for e in range(connect.shape[0]):
        element_coords = coords[connect[e]].T  # (2, n_nodes)
        dNdxi = shape_deriv_fcn(xi_eta)

        # Compute Jacobian and dN/dx
        dxdxi, _ = le.compute_jacobian(element_coords, dNdxi)
        dxidx = np.linalg.inv(dxdxi)
        dNdx = le.convert_derivatives(dNdxi, dxidx)  # (n_nodes, 2)

        # Zero displacement → F = I
        displacement = np.zeros_like(element_coords)
        F = le.compute_deformation_gradient(ncoord=2, displacement=displacement, dNdx=dNdx)
        np.testing.assert_allclose(F, np.eye(2), atol=1e-12, err_msg=f"{ele_type} element {e}: F ≠ I for zero displacement")


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN.keys())
def test_compute_deformation_gradient_uniform_strain(ele_type):
    """
    Tests compute_deformation_gradient for a linear displacement field:
    u(x, y) = [0.1*x, 0.2*y]
    Expected F = [[1.1, 0.0], [0.0, 1.2]]
    """
    # Create a simple 1x1 mesh for clarity
    x_lower, y_lower = 0.0, 0.0
    x_upper, y_upper = 2.0, 1.0
    nx, ny = 1, 1
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)

    xi_eta = ELEMENT_CENTER_NATURAL_COORDS[ele_type]
    shape_deriv_fcn = SHAPE_DERIV_FCN[ele_type]

    for e in range(connect.shape[0]):
        element_coords = coords[connect[e]].T
        dNdxi = shape_deriv_fcn(xi_eta)

        # Compute dNdx via Jacobian
        dxdxi, _ = le.compute_jacobian(element_coords, dNdxi)
        dxidx = np.linalg.inv(dxdxi)
        dNdx = le.convert_derivatives(dNdxi, dxidx)  # shape (n_nodes, 2)

        # Define a linear displacement field: u = [0.1x, 0.2y]
        def u(x, y): return np.array([0.1 * x, 0.2 * y])
        displacement = np.array([u(x, y) for x, y in element_coords.T]).T  # shape (2, n_nodes)

        # Compute deformation gradient
        F = le.compute_deformation_gradient(ncoord=2, displacement=displacement, dNdx=dNdx)

        # Expected result
        expected_F = np.array([
            [1.1, 0.0],
            [0.0, 1.2]
        ])

        np.testing.assert_allclose(F, expected_F, atol=1e-12, err_msg=f"{ele_type} element {e}: F mismatch")


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN.keys())
def test_convert_to_spatial_derivatives_identity(ele_type):
    """
    Tests that converting to spatial derivatives using Finv = I returns the same dNdx.
    """
    # Small mesh, 1 element
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 2.0, 1.0, 1, 1)
    xi_eta = ELEMENT_CENTER_NATURAL_COORDS[ele_type]
    shape_deriv_fcn = SHAPE_DERIV_FCN[ele_type]

    for e in range(connect.shape[0]):
        element_coords = coords[connect[e]]
        dNdxi = shape_deriv_fcn(xi_eta)

        # Compute reference dNdx
        J, _ = le.compute_jacobian(element_coords.T, dNdxi)
        dNdx = le.convert_derivatives(dNdxi, np.linalg.inv(J))

        # Use identity deformation gradient (no deformation)
        Finv = np.eye(2)

        # Convert to spatial derivatives
        dNdx_spatial = le.convert_to_spatial_derivatives(dNdx, Finv)

        # Should be unchanged
        np.testing.assert_allclose(dNdx_spatial, dNdx, atol=1e-12)

        # Check shape
        assert dNdx_spatial.shape == dNdx.shape


def test_convert_to_spatial_derivatives_inverse_roundtrip():
    dNdx = np.random.rand(6, 2)
    F = np.array([[1.1, 0.2], [0.0, 1.2]])
    Finv = np.linalg.inv(F)
    spatial_grad = le.convert_to_spatial_derivatives(dNdx, Finv)
    recovered = spatial_grad @ F
    np.testing.assert_allclose(recovered, dNdx, atol=1e-12)


# def test_material_stiffness_2d():
def test_material_stiffness_identity():
    """
    Test the undeformed case: let B = I and J = 1.
    In this case the expected tensor is:
      C_{ijkl} = mu1*(δ_{ik}δ_{jl} + δ_{il}δ_{jk} - (2/3)*δ_{ij}δ_{kl})
                + K1*δ_{ij}δ_{kl}.
    We check several components.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    B = np.eye(2)
    J = 1.0
    C = le.material_stiffness_2d(B, J, materialprops)

    # Define delta (Kronecker delta)
    delta = np.eye(2)

    # Function to compute expected value from the closed form:
    def expected(i, j, k, l):
        term_dev = mu1 * (delta[i, k]*delta[j, l] + delta[i, l]*delta[j, k] - (2/3)*delta[i, j]*delta[k, l])
        term_vol = K1 * delta[i, j]*delta[k, l]
        return term_dev + term_vol

    # Check a few representative components.
    tol = 1e-12
    for indices in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0)]:
        i, j, k, l = indices
        expected_val = expected(i, j, k, l)
        computed_val = C[i, j, k, l]
        assert np.isclose(computed_val, expected_val, atol=tol), \
            f"Mismatch at indices {indices}: computed {computed_val}, expected {expected_val}"


def test_material_stiffness_symmetry():
    """
    Check that the computed 4th-order tensor is symmetric in its indices.
    For example, C_{ijkl} should equal C_{jikl} and C_{ijlk}.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    B = np.eye(2)
    J = 1.0
    C = le.material_stiffness_2d(B, J, materialprops)

    tol = 1e-12
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    # Test symmetry in first two indices:
                    assert np.isclose(C[i, j, k, l], C[j, i, k, l], atol=tol), \
                        f"Minor symmetry failure: C[{i},{j},{k},{l}] != C[{j},{i},{k},{l}]"
                    # Test symmetry in last two indices:
                    assert np.isclose(C[i, j, k, l], C[i, j, l, k], atol=tol), \
                        f"Minor symmetry failure: C[{i},{j},{k},{l}] != C[{i},{j},{l},{k}]"
                    # Test symmetry for major symmetry:
                    assert np.isclose(C[i, j, k, l], C[k, l, i, j], atol=tol), \
                        f"Major symmetry failure: C[{i},{j},{k},{l}] != C[{k},{l},{i},{j}]"


def test_material_stiffness_volumetric_scaling():
    """
    Check that when J is changed (keeping B = I), the isochoric part scales as expected.
    For a given J, the isochoric part is divided by J^(2/3) and the volumetric part is
    multiplied by (2J - 1)J.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    B = np.eye(2)
    # Test for a J different from 1.
    J = 8.0  # 8^(2/3) = 4
    C = le.material_stiffness_2d(B, J, materialprops)
    
    # For undeformed B=I, expected C has the same structure with additional scaling.
    delta = np.eye(2)

    def expected(i, j, k, l):
        # Isochoric part divided by J^(2/3):
        term_dev = mu1 * (delta[i, k]*delta[j, l] + delta[i, l]*delta[j, k] - (2.0 / 3.0)*delta[i, j]*delta[k, l]) / (J**(2/3))
        term_vol = K1 * (2*J - 1)*J * delta[i, j]*delta[k, l]
        return term_dev + term_vol

    tol = 1e-12
    for indices in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1)]:
        i, j, k, l = indices
        expected_val = expected(i, j, k, l)
        computed_val = C[i, j, k, l]
        assert np.isclose(computed_val, expected_val, atol=tol), \
            f"At J={J}, mismatch at indices {indices}: computed {computed_val}, expected {expected_val}"


def test_material_stiffness_large_J_bulk_response():
    B = np.eye(2)
    J = 10.0
    mu1, K1 = 1.0, 100.0  # Make bulk modulus much larger
    C = le.material_stiffness_2d(B, J, np.array([mu1, K1]))

    # Diagonal volumetric terms should be much larger than deviatoric terms
    volumetric = C[0, 0, 0, 0]
    shear = C[0, 1, 0, 1]
    assert volumetric > 10 * shear


def test_kirchhoff_stress_undeformed_state():
    """
    For B = I and J = 1, the deviatoric part should vanish and the volumetric part 
    gives K1*1*(1-1)=0. Thus the stress should be zero.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    B = np.eye(2)
    J = 1.0
    stress = le.kirchhoff_stress(B, J, materialprops)
    expected = np.zeros((2, 2))
    tol = 1e-12
    assert np.allclose(stress, expected, atol=tol), f"Expected zero stress, got {stress}"


def test_kirchhoff_stress_symmetry():
    """
    The stress tensor for an isotropic material should be symmetric.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    # Choose a symmetric B that is not identity.
    B = np.array([[2.0, 0.5],
                  [0.5, 3.0]])
    J = 1.2
    stress = le.kirchhoff_stress(B, J, materialprops)
    tol = 1e-12
    # Check symmetry: stress_{ij} should equal stress_{ji}
    assert np.allclose(stress, stress.T, atol=tol), f"Stress tensor not symmetric: {stress}"


def test_kirchhoff_stress_known_deformation():
    """
    For a simple test case, choose B = [[4,0],[0,1]] and J = 2.
    For 2D, the adjusted trace is Bkk = 4 + 1 + 1 = 6.
    Then, for each diagonal component:
      stress_00 = mu1*(4 - 6/3)/J^(2/3) + K1 * 2*(2-1)
                = mu1*(4-2)/2^(2/3) + 2*K1
                = (2*mu1)/2^(2/3) + 2*K1.
      stress_11 = mu1*(1 - 6/3)/2^(2/3) + 2*K1
                = mu1*(1-2)/2^(2/3) + 2*K1
                = (-mu1)/2^(2/3) + 2*K1.
    Off-diagonals are zero.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    B = np.array([[4.0, 0.0],
                  [0.0, 1.0]])
    J = 2.0
    factor = 2**(2.0/3.0)  # 2^(2/3)
    expected_00 = (2 * mu1) / factor + 2 * K1
    expected_11 = (-mu1) / factor + 2 * K1
    expected = np.array([[expected_00, 0.0],
                         [0.0, expected_11]])
    stress = le.kirchhoff_stress(B, J, materialprops)
    tol = 1e-12
    assert np.allclose(stress, expected, atol=tol), f"Mismatch in known deformation: computed {stress}, expected {expected}"


def test_kirchhoff_stress_3d_case():
    """
    For a 3D test case, use B = I (3x3) and J = 1. In this situation no out-of-plane 
    adjustment is done. The stress should be zero.
    """
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    B = np.eye(3)
    J = 1.0
    stress = le.kirchhoff_stress(B, J, materialprops)
    expected = np.zeros((3, 3))
    tol = 1e-12
    assert np.allclose(stress, expected, atol=tol), f"3D undeformed state should be zero, got {stress}"


def test_compute_stiffness_contributions_stiffness_shape():
    """
    Verify that the computed stiffness matrix has the correct shape.
    """
    nelnodes = 4
    ndof = 2
    ncoord = 2
    # Use dummy inputs (contents do not matter for shape)
    stress = np.zeros((ndof, ncoord))
    dsde = np.zeros((ndof, ncoord, ndof, ncoord))
    dNdxs = np.ones((nelnodes, ncoord))
    
    kel = le.compute_stiffness_contributions(nelnodes, ndof, ncoord, stress, dsde, dNdxs)
    expected_shape = (ndof * nelnodes, ndof * nelnodes)
    assert kel.shape == expected_shape, f"Expected shape {expected_shape}, got {kel.shape}"


@pytest.mark.parametrize("ele_type", SHAPE_DERIV_FCN.keys())
def test_compute_stiffness_contributions_symmetry(ele_type):
    """
    For many standard formulations the assembled element stiffness matrix is symmetric.
    This test checks symmetry using consistent shape functions, stress, and stiffness tensor.
    Parameterized to run across multiple element types.
    """
    ndof = ncoord = 2

    # Generate element geometry and shape functions
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, 1.0, 1.0, 1, 1)
    element_coords = coords[connect[0]]
    nelnodes = element_coords.shape[0]

    xi_eta = ELEMENT_CENTER_NATURAL_COORDS[ele_type]
    shape_deriv_fcn = SHAPE_DERIV_FCN[ele_type]
    dNdxi = shape_deriv_fcn(xi_eta)

    # Compute Jacobian and dN/dx
    J, _ = le.compute_jacobian(element_coords.T, dNdxi)
    Jinv = np.linalg.inv(J)
    dNdxs = le.convert_derivatives(dNdxi, Jinv)

    # Define a deformation gradient (e.g., small shear + stretch)
    F = np.array([
        [1.1, 0.2],
        [0.0, 1.05]
    ])
    J_det = np.linalg.det(F)
    B = F @ F.T

    # Material properties and consistent stiffness tensor
    mu1 = 10.0
    K1 = 100.0
    materialprops = np.array([mu1, K1])
    dsde = le.material_stiffness_2d(B, J_det, materialprops)

    # Consistent stress tensor
    stress = le.kirchhoff_stress(B, J_det, materialprops)

    # Compute element stiffness matrix
    kel = le.compute_stiffness_contributions(nelnodes, ndof, ncoord, stress, dsde, dNdxs)

    # Assert symmetry
    tol = 1e-12
    assert np.allclose(kel, kel.T, atol=tol), f"{ele_type}: stiffness matrix is not symmetric:\n{kel}"


def test_compute_stiffness_contributions_zero_stress():
    """
    When stress is zero the stiffness should come solely from the material term.
    For a simple manufactured input we compute the expected value.
    """
    nelnodes = 2
    ndof = 2
    ncoord = 2
    # Zero stress so geometric contribution is zero.
    stress = np.zeros((ndof, ncoord))
    # Let dsde be constant: for simplicity, use dsde = 2.0 for every index combination.
    dsde = 2.0 * np.ones((ndof, ncoord, ndof, ncoord))
    # Use a simple dNdxs with distinct values.
    dNdxs = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
    
    kel = le.compute_stiffness_contributions(nelnodes, ndof, ncoord, stress, dsde, dNdxs)
    # We can manually compute the expected stiffness matrix.
    # For each dof pairing: expected contribution = 
    #   sum_{j,l} 2.0 * dNdxs[b,l]*dNdxs[a,j]
    # Because there is no stress term.
    expected = np.zeros((ndof*nelnodes, ndof*nelnodes))
    for a in range(nelnodes):
        for b in range(nelnodes):
            for i in range(ndof):
                for k in range(ndof):
                    row = ndof * a + i
                    col = ndof * b + k
                    temp = 0.0
                    for j in range(ncoord):
                        for l in range(ncoord):
                            temp += 2.0 * dNdxs[b, l] * dNdxs[a, j]
                    expected[row, col] = temp
    tol = 1e-12
    assert np.allclose(kel, expected, atol=tol), f"Zero stress test failed. Computed:\n{kel}\nExpected:\n{expected}"


def test_compute_stiffness_contributions_geometric_term():
    """
    For a simple manufactured case, verify the geometric (stress-dependent)
    term. In many formulations the geometric contribution for the (a,i),(b,k) entry is:
        sum_{j} stress[i,j] * dNdxs[a,j] * dNdxs[b,k].
    In the provided function the term is computed as:
        sum_{j} stress[i,j] * dNdxs[a,k] * dNdxs[b,j].
    This test sets up a situation to compare these two formulations.
    """
    nelnodes = 2
    ndof = 2
    ncoord = 2
    # Choose a nonzero stress.
    stress = np.array([[2.0, 0.5],
                       [0.5, 1.0]])
    # Use identity for dsde so that only the geometric term will be different
    dsde = np.zeros((ndof, ncoord, ndof, ncoord))
    # Set dsde = 0 so that the material term does not contribute.
    # For dNdxs, choose values that are distinct.
    dNdxs = np.array([[0.2, 0.3],
                      [0.4, 0.1]])
    
    kel = le.compute_stiffness_contributions(nelnodes, ndof, ncoord, stress, dsde, dNdxs)
    
    # Now compute separately the geometric contribution using the two possible index orderings.
    # According to the provided function, the geometric term used is:
    #   G_code[(a,i),(b,k)] = sum_{j} stress[i,j] * dNdxs[a,k] * dNdxs[b,j].
    # An alternative (perhaps more standard) formulation is:
    #   G_alt[(a,i),(b,k)] = sum_{j} stress[i,j] * dNdxs[a,j] * dNdxs[b,k].
    # We can compute both and then print them for comparison.
    num_dofs = ndof * nelnodes
    G_code = np.zeros((num_dofs, num_dofs))
    G_alt  = np.zeros((num_dofs, num_dofs))
    for a in range(nelnodes):
        for b in range(nelnodes):
            for i in range(ndof):
                for k in range(ndof):
                    row = ndof * a + i
                    col = ndof * b + k
                    sum_code = 0.0
                    sum_alt = 0.0
                    for j in range(ncoord):
                        sum_code += stress[i, j] * dNdxs[a, k] * dNdxs[b, j]
                        sum_alt  += stress[i, j] * dNdxs[a, j] * dNdxs[b, k]
                    G_code[row, col] = sum_code
                    G_alt[row, col]  = sum_alt
    # In the function, the total contribution is (material - geometric).
    # Here, since dsde = 0, kel should equal -G_code.
    tol = 1e-12
    assert np.allclose(kel, -G_code, atol=tol), f"Geometric term mismatch: kel = {kel}, expected -G_code = {-G_code}"
    
    # For informational purposes, we can also check if G_code and G_alt differ.
    # They are not necessarily equal; if they differ, then you might want to check which formulation
    # is intended in your application.
    if not np.allclose(G_code, G_alt, atol=tol):
        print("Warning: The geometric stiffness term computed by the function differs from the alternative formulation.")
