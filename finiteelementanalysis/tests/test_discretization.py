from finiteelementanalysis import discretization as di
import numpy as np
import pytest


###########################################################
#  TEST WRAPPER FUNCTIONS
###########################################################
@pytest.mark.parametrize("ele_type, expected", [
    ("D1_nn2",       (1, 1, 2)),
    ("D1_nn3",       (1, 1, 3)),
    ("D2_nn3_tri",   (2, 2, 3)),
    ("D2_nn6_tri",   (2, 2, 6)),
    ("D2_nn4_quad",  (2, 2, 4)),
    ("D2_nn8_quad",  (2, 2, 8)),
])
def test_element_info_known(ele_type, expected):
    """
    Test that known element types return the expected (ncoord, ndof, nelnodes).
    """
    result = di.element_info(ele_type)
    assert result == expected, f"For {ele_type}, expected {expected} but got {result}"


def test_element_info_unknown():
    """
    Test that an unrecognized element type raises ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        di.element_info("D3_unknown_hex")
    assert "Unknown element type" in str(excinfo.value), \
        "Expected ValueError for unrecognized element type."


@pytest.mark.parametrize("ele_type, xi, expected_size", [
    ("D1_nn2", np.array([0.0]), 2),   # 1D, 2-node
    ("D1_nn3", np.array([0.0]), 3),   # 1D, 3-node
    ("D2_nn3_tri", np.array([0.33, 0.33]), 3),  # 2D triangle (3 nodes)
    ("D2_nn6_tri", np.array([0.5, 0.2]), 6),   # 2D quadratic triangle (6 nodes)
    ("D2_nn4_quad", np.array([0.0, 0.0]), 4),  # 2D quad (4 nodes)
    ("D2_nn8_quad", np.array([0.25, 0.25]), 8),  # 2D quad (8 nodes)
])
def test_shape_fcn_known(ele_type, xi, expected_size):
    """
    Test shape_fcn on known element types at a sample xi,
    verifying that the returned array has the correct size.
    Does NOT check if shape functions themselves are correct - that is done in other tests.
    """
    N = di.shape_fcn(ele_type, xi)
    assert isinstance(N, np.ndarray), "shape_fcn must return a numpy array."
    assert N.shape == (expected_size, 1), (
        f"For {ele_type} at xi={xi}, expected shape ({expected_size},) but got {N.shape}."
    )


def test_shape_fcn_unknown():
    """
    Test that an unsupported element type raises ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        di.shape_fcn("non_existent_type", np.array([0.0]))
    assert "Unsupported element type" in str(excinfo.value), (
        "Expected ValueError for unknown element type."
    )


@pytest.mark.parametrize("ele_type, xi, expected_shape", [
    # 1D elements (derivatives wrt 1 param => shape=(n_nodes, 1))
    ("D1_nn2",       np.array([0.0]),        (2, 1)),
    ("D1_nn3",       np.array([0.0]),        (3, 1)),
    # 2D elements (derivatives wrt 2 params => shape=(n_nodes, 2))
    ("D2_nn3_tri",   np.array([0.33, 0.33]), (3, 2)),
    ("D2_nn6_tri",   np.array([0.2,  0.4]),  (6, 2)),
    ("D2_nn4_quad",  np.array([0.0,  0.0]),  (4, 2)),
    ("D2_nn8_quad",  np.array([0.25, 0.25]), (8, 2)),
])
def test_shape_fcn_derivative_known(ele_type, xi, expected_shape):
    """
    Test shape_fcn_derivative on known element types at a sample xi,
    verifying that the returned array has the correct shape 
    (n_nodes, n_deriv_params).
    """
    dN_dxi = di.shape_fcn_derivative(ele_type, xi)
    assert isinstance(dN_dxi, np.ndarray), "shape_fcn_derivative must return a numpy array."
    assert dN_dxi.shape == expected_shape, (
        f"For {ele_type} at xi={xi}, expected shape {expected_shape} but got {dN_dxi.shape}."
    )


def test_shape_fcn_derivative_unknown():
    """
    Test that an unsupported element type raises ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        di.shape_fcn_derivative("non_existent_type", np.array([0.0]))
    assert "Unsupported element type" in str(excinfo.value), (
        "Expected ValueError for unknown element type."
    )


@pytest.mark.parametrize("ele_type, expected_npts, dim", [
    ("D1_nn2",       2, 1),  # 1D, 2-pt integration
    ("D1_nn3",       3, 1),  # 1D, 3-pt integration
    ("D2_nn3_tri",   1, 2),  # 2D triangle, 1-pt
    ("D2_nn6_tri",   3, 2),  # 2D triangle, 3-pt
    ("D2_nn4_quad",  4, 2),  # 2D quad, 4-pt
    ("D2_nn8_quad",  9, 2),  # 2D quad, 9-pt
])
def test_integration_info_known(ele_type, expected_npts, dim):
    """
    Test integration_info for known element types, verifying that the returned 
    number of points, points array, and weights array match expectations.
    """
    npts, points, weights = di.integration_info(ele_type)

    # Check number of points
    assert npts == expected_npts, (
        f"Expected {expected_npts} integration points for {ele_type}, got {npts}."
    )

    # Check shape of points array
    assert points.shape[1] == npts, (
        f"points should have {npts} columns. Instead got shape {points.shape}."
    )
    if dim == 1:
        assert points.shape[0] == 1, f"1D points array shape mismatch: {points.shape}"
    elif dim == 2:
        assert points.shape[0] == 2, f"2D points must have 2 dims. shape={points.shape}"

    # Check shape of weights array
    assert weights.shape[0] == npts, (
        f"Weights should have {npts} entries. Instead got shape {weights.shape}."
    )
    # Typically shape is (npts,) or (npts,1). We'll just ensure length is npts.
    # If you want a stricter check (like must be shape (npts,)), do that here.


def test_integration_info_unknown():
    """
    Test that an unrecognized element type raises ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        di.integration_info("D3_nonexistent_hex")
    assert "Unknown element type" in str(excinfo.value), (
        "Expected ValueError for an unrecognized element type."
    )


@pytest.mark.parametrize("ele_type, face, expected_face_type, expected_num_nodes, expected_face_nodes", [
    # 1D line with 2 nodes => 2 "faces" (endpoints)
    ("D1_nn2", 0, None, 1, [0]),
    ("D1_nn2", 1, None, 1, [1]),

    # 2D triangle (3 nodes), each edge is 2 corner nodes => 'face_elem_type="D1_nn2"'
    ("D2_nn3_tri", 0, "D1_nn2", 2, [0,1]),
    ("D2_nn3_tri", 1, "D1_nn2", 2, [1,2]),
    ("D2_nn3_tri", 2, "D1_nn2", 2, [2,0]),

    # 2D 4-node quad => each edge 2 corner nodes => 'face_elem_type="D1_nn2"'
    ("D2_nn4_quad", 0, "D1_nn2", 2, [0,1]),
    ("D2_nn4_quad", 1, "D1_nn2", 2, [1,2]),
    ("D2_nn4_quad", 2, "D1_nn2", 2, [2,3]),
    ("D2_nn4_quad", 3, "D1_nn2", 2, [3,0]),
])
def test_face_info_valid(
    ele_type, face, expected_face_type, expected_num_nodes, expected_face_nodes
):
    """
    Test valid element types and valid face indices, checking the returned
    face_element_type, num_face_nodes, and nodes_on_face.
    """
    face_elem_type, num_face_nodes, nodes_on_face = di.face_info(ele_type, face)

    assert face_elem_type == expected_face_type, (
        f"Expected face_elem_type={expected_face_type} for {ele_type}, face={face}, "
        f"but got {face_elem_type}."
    )
    assert num_face_nodes == expected_num_nodes, (
        f"Expected num_face_nodes={expected_num_nodes} for {ele_type}, face={face}, "
        f"but got {num_face_nodes}."
    )
    assert nodes_on_face == expected_face_nodes, (
        f"Expected nodes_on_face={expected_face_nodes} for {ele_type}, face={face}, "
        f"but got {nodes_on_face}."
    )


def test_face_info_out_of_range_face():
    """
    Test that an out-of-range face index raises an error.
    """
    ele_type = "D2_nn4_quad"
    invalid_face = 99  # definitely out of range

    # We can check for ValueError or RuntimeError, whichever your code does.
    # Example: your code might raise ValueError, or might fail the 'len mismatch' => RuntimeError
    with pytest.raises((ValueError, RuntimeError)):
        di.face_info(ele_type, invalid_face)


def test_face_info_unknown_element():
    """
    Test that an unrecognized element type raises ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        di.face_info("D9_unknown_element", 0)
    assert "Unknown element type" in str(excinfo.value), (
        "Expected ValueError for unknown element type."
    )


###########################################################
#  TEST SHAPE FUNCTIONS AND DERIVATIVES
###########################################################

@pytest.mark.parametrize("xi, expected", [
    (np.array([-1.0]), np.array([[1.0], [0.0]])),  # N1=1, N2=0 at xi=-1
    (np.array([1.0]), np.array([[0.0], [1.0]])),  # N1=0, N2=1 at xi=1
    (np.array([0.0]), np.array([[0.5], [0.5]])),  # N1=0.5, N2=0.5 at xi=0 (midpoint)
    (np.array([-0.5]), np.array([[0.75], [0.25]])),  # N1=0.75, N2=0.25 at xi=-0.5
    (np.array([0.5]), np.array([[0.25], [0.75]])),  # N1=0.25, N2=0.75 at xi=0.5
])
def test_D1_nn2(xi, expected):
    """
    Test the D1_nn2 function, verifying:
      - The returned array has shape (2,1)
      - The shape function values match expected results (within tolerance).
    """
    result = di.D1_nn2(xi)
    # Check shape
    assert result.shape == (2, 1), f"Expected shape (2,1) but got {result.shape}."
    # Check values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


@pytest.mark.parametrize("xi", [
    np.array([0.0]),
    np.array([1.0]),
    np.array([-1.0]),
    np.array([0.25]),
])
def test_D1_nn2_dxi(xi):
    """
    Test D1_nn2_dxi at various xi points, ensuring that:
      - The returned array has shape (2,1),
      - The derivative values are [0.5, -0.5] (constant).
    """
    dN = di.D1_nn2_dxi(xi)
    # Check shape
    assert dN.shape == (2, 1), f"Expected shape (2,1), got {dN.shape}."

    # Check values: dN1/dxi= -0.5, dN2/dxi= 0.5
    expected = np.array([[-0.5], [0.5]])
    assert np.allclose(dN, expected, atol=1e-12), (
        f"For xi={xi}, expected derivatives {expected.ravel()} but got {dN.ravel()}."
    )


@pytest.mark.parametrize("xi, expected", [
    # xi=0 => N1=0, N2=0, N3=1
    (np.array([0.0]), np.array([[0.0], [0.0], [1.0]])),
    # xi=1 => N1=0, N2=1, N3=0
    (np.array([1.0]), np.array([[0.0], [1.0], [0.0]])),
    # xi=-1 => N1=1, N2=0, N3=0
    (np.array([-1.0]), np.array([[1.0], [0.0], [0.0]])),
    # an intermediate point, e.g. xi=0.5 => 
    #   N1(0.5) = -0.125, N2(0.5) = 0.375, N3(0.5) = 0.75
    (np.array([0.5]), np.array([[-0.125], [0.375], [0.75]])),
])
def test_D1_nn3(xi, expected):
    """
    Test the D1_nn3 function at various xi points, verifying:
      - The returned array has shape (3,1)
      - The shape function values match expected results (within tolerance).
    """
    N = di.D1_nn3(xi)
    # Check shape
    assert N.shape == (3, 1), f"Expected shape (3,1) but got {N.shape}."
    # Check values
    assert np.allclose(N, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {N.ravel()}."
    )


@pytest.mark.parametrize("xi, expected", [
    # xi=-1 => dN1/dxi=-1.5, dN2/dxi=-0.5, dN3/dxi=2.0
    (np.array([-1.0]), np.array([[-1.5], [-0.5], [2.0]])),
    # xi=0 => dN1/dxi=-0.5, dN2/dxi=0.5, dN3/dxi=0.0
    (np.array([0.0]), np.array([[-0.5], [0.5], [0.0]])),
    # xi=1 => dN1/dxi=0.5, dN2/dxi=1.5, dN3/dxi=-2.0
    (np.array([1.0]), np.array([[0.5], [1.5], [-2.0]])),
    # xi=0.5 => dN1/dxi=0.0, dN2/dxi=1.0, dN3/dxi=-1.0
    (np.array([0.5]), np.array([[0.0], [1.0], [-1.0]])),
    # xi=-0.5 => dN1/dxi=-1.0, dN2/dxi=0.0, dN3/dxi=1.0
    (np.array([-0.5]), np.array([[-1.0], [0.0], [1.0]])),
    # Additional test cases for robustness
    (np.array([0.25]), np.array([[-0.25], [0.75], [-0.5]])),
    (np.array([-0.25]), np.array([[-0.75], [0.25], [0.5]])),
])
def test_D1_nn3_dxi(xi, expected):
    """
    Test the D1_nn3_dxi function at various xi points, verifying:
      - The returned array has shape (3,1)
      - The derivative values match expected results (within tolerance).
    """
    result = di.D1_nn3_dxi(xi)
    # Check shape
    assert result.shape == (3, 1), f"Expected shape (3,1) but got {result.shape}."
    # Check values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


@pytest.mark.parametrize("xi, expected", [
    # (ξ, η) = (0,0) => N1=0, N2=0, N3=1
    (np.array([0.0, 0.0]), np.array([[0.0], [0.0], [1.0]])),
    # (ξ, η) = (1,0) => N1=1, N2=0, N3=0
    (np.array([1.0, 0.0]), np.array([[1.0], [0.0], [0.0]])),
    # (ξ, η) = (0,1) => N1=0, N2=1, N3=0
    (np.array([0.0, 1.0]), np.array([[0.0], [1.0], [0.0]])),
    # (ξ, η) = (0.5, 0.5) => N1=0.5, N2=0.5, N3=0
    (np.array([0.5, 0.5]), np.array([[0.5], [0.5], [0.0]])),
    # (ξ, η) = (0.5, 0.0) => N1=0.5, N2=0.0, N3=0.5
    (np.array([0.5, 0.0]), np.array([[0.5], [0.0], [0.5]])),
    # (ξ, η) = (0.0, 0.5) => N1=0.0, N2=0.5, N3=0.5
    (np.array([0.0, 0.5]), np.array([[0.0], [0.5], [0.5]])),
    # (ξ, η) = (1/3, 1/3) => N1=1/3, N2=1/3, N3=1/3 (centroid of triangle)
    (np.array([1/3, 1/3]), np.array([[1/3], [1/3], [1/3]])),
])
def test_D2_nn3_tri(xi, expected):
    """
    Test the D2_nn3_tri function at various (ξ, η) points, verifying:
      - The returned array has shape (3,1)
      - The shape function values match expected results (within tolerance).
    """
    result = di.D2_nn3_tri(xi)
    # Check shape
    assert result.shape == (3, 1), f"Expected shape (3,1) but got {result.shape}."
    # Check values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


@pytest.mark.parametrize("xi, expected", [
    # The derivatives are constant, so any (ξ, η) should return the same result
    (np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
    (np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
    (np.array([0.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
    (np.array([0.5, 0.5]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
    (np.array([1/3, 1/3]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
])
def test_D2_nn3_tri_dxi(xi, expected):
    """
    Test the D2_nn3_tri_dxi function, verifying:
      - The returned array has shape (3,2)
      - The derivative values are correct and constant.
    """
    result = di.D2_nn3_tri_dxi(xi)
    # Check shape
    assert result.shape == (3, 2), f"Expected shape (3,2) but got {result.shape}."
    # Check values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


@pytest.mark.parametrize("xi, expected", [
    # Corner node 1 => (ξ, η) = (1, 0) => shape function #1 =1, others=0
    (np.array([1.0, 0.0]), np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])),
    # Corner node 2 => (ξ, η) = (0, 1) => shape function #2 =1, others=0
    (np.array([0.0, 1.0]), np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])),
    # Corner node 3 => (ξ, η) = (0, 0) => shape function #3 =1, others=0
    (np.array([0.0, 0.0]), np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])),
    # Mid-side node 4 => (ξ, η) = (0.5, 0.5) => shape function #4 =1, others=0
    (np.array([0.5, 0.5]), np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])),
    # Mid-side node 5 => (ξ, η) = (0, 0.5) => shape function #5 =1, others=0
    (np.array([0.0, 0.5]), np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]])),
    # Mid-side node 6 => (ξ, η) = (0.5, 0) => shape function #6 =1, others=0
    (np.array([0.5, 0.0]), np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])),
    # A centroid example => (ξ, η) = (1/3, 1/3)
    # If you want exact symbolic, you can compute from the definitions or skip.
    (np.array([1/3, 1/3]), np.array([[-0.11111111],  # N1
                                     [-0.11111111],  # N2
                                     [-0.11111111],  # N3
                                     [0.44444444],  # N4
                                     [0.44444444],  # N5
                                     [0.44444444]]))  # N6
])
def test_D2_nn6_tri(xi, expected):
    """
    Test the D2_nn6_tri function at various (xi, η) points in the reference triangle,
    verifying:
      - The returned array has shape (6,1).
      - The shape function values match the expected results (within tolerance).
    """
    result = di.D2_nn6_tri(xi)

    # Check shape
    assert result.shape == (6, 1), f"Expected shape (6,1) but got {result.shape}."

    # Check numeric values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


def reference_d2_nn6_tri_dxi(xi, eta):
    """
    Computes the reference derivatives of the 2D quadratic shape functions
    for a six-node triangular element.
    """
    xic = 1.0 - xi - eta  # Complementary coordinate ξ_c
    dN_ref = np.zeros((6, 2))
    
    # Corner nodes (primary nodes)
    dN_ref[0] = [4.0 * xi - 1.0, 0.0]  # ∂N1/∂ξ, ∂N1/∂η
    dN_ref[1] = [0.0, 4.0 * eta - 1.0]  # ∂N2/∂ξ, ∂N2/∂η
    dN_ref[2] = [-(4.0 * xic - 1.0), -(4.0 * xic - 1.0)]  # ∂N3/∂ξ, ∂N3/∂η

    # Mid-edge nodes
    dN_ref[3] = [4.0 * eta, 4.0 * xi]  # ∂N4/∂ξ, ∂N4/∂η
    dN_ref[4] = [-4.0 * eta, 4.0 * (xic - eta)]  # ∂N5/∂ξ, ∂N5/∂η
    dN_ref[5] = [4.0 * (xic - xi), -4.0 * xi]  # ∂N6/∂ξ, ∂N6/∂η
    
    return dN_ref

@pytest.mark.parametrize("xi_vec", [
    # corners
    np.array([1.0, 0.0]),     # Node 1
    np.array([0.0, 1.0]),     # Node 2
    np.array([0.0, 0.0]),     # Node 3
    # mid-side
    np.array([0.5, 0.5]),     # Node 4
    np.array([0.0, 0.5]),     # Node 5
    np.array([0.5, 0.0]),     # Node 6
    # centroid
    np.array([1.0/3.0, 1.0/3.0]),
])
def test_D2_nn6_tri_dxi(xi_vec):
    """
    Test the D2_nn6_tri_dxi function at various (xi, eta) points,
    ensuring:
      - The returned array has shape (6,2).
      - The derivative values match the known formulas (within tolerance).
    """
    # Evaluate the function under test
    dN_num = di.D2_nn6_tri_dxi(xi_vec)
    # Evaluate a reference version
    xi, eta = xi_vec[0], xi_vec[1]
    dN_ref = reference_d2_nn6_tri_dxi(xi, eta)

    # Check shape
    assert dN_num.shape == (6, 2), f"Expected shape (6,2), got {dN_num.shape}."

    # Compare numeric values
    assert np.allclose(dN_num, dN_ref, atol=1e-12), (
        f"For xi={xi_vec}, got differences:\n{dN_num - dN_ref}"
    )


@pytest.mark.parametrize("xi, expected", [
    # (ξ, η) = (-1,-1) => N1=1, others 0
    (np.array([-1.0, -1.0]), np.array([[1.0], [0.0], [0.0], [0.0]])),
    # (ξ, η) = (1,-1) => N2=1, others 0
    (np.array([1.0, -1.0]), np.array([[0.0], [1.0], [0.0], [0.0]])),
    # (ξ, η) = (1,1) => N3=1, others 0
    (np.array([1.0, 1.0]), np.array([[0.0], [0.0], [1.0], [0.0]])),
    # (ξ, η) = (-1,1) => N4=1, others 0
    (np.array([-1.0, 1.0]), np.array([[0.0], [0.0], [0.0], [1.0]])),
    # (ξ, η) = (0,0) => centroid (should be all 0.25)
    (np.array([0.0, 0.0]), np.array([[0.25], [0.25], [0.25], [0.25]])),
    # (ξ, η) = (0.5,0.5) => intermediate point (computed manually)
    (np.array([0.5, 0.5]), np.array([[0.0625], [0.1875], [0.5625], [0.1875]])),
])
def test_D2_nn4_quad(xi, expected):
    """
    Test the D2_nn4_quad function, verifying:
      - The returned array has shape (4,1)
      - The shape function values match expected results (within tolerance).
    """
    result = di.D2_nn4_quad(xi)
    # Check shape
    assert result.shape == (4, 1), f"Expected shape (4,1) but got {result.shape}."
    # Check values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


@pytest.mark.parametrize("xi_vec, expected", [
    # Corner (ξ=-1, η=-1)
    (
        np.array([-1.0, -1.0]),
        np.array([
            [-0.5, -0.5],
            [ 0.5, 0.0],
            [ 0.0,  0.0 ],
            [ 0.0,  0.5 ]
        ])
    ),
    # Corner (ξ=1, η=-1)
    (
        np.array([1.0, -1.0]),
        np.array([
            [-0.5,  0.0 ],
            [ 0.5, -0.5 ],
            [ 0.0,  0.5 ],
            [ 0.0,  0.0 ]
        ])
    ),
    # Corner (ξ=1, η=1)
    (
        np.array([1.0,  1.0]),
        np.array([
            [ 0.0,  0.0 ],
            [ 0.0, -0.5 ],
            [ 0.5,  0.5 ],
            [-0.5,  0.0 ]
        ])
    ),
    # Corner (ξ=-1, η=1)
    (
        np.array([-1.0,  1.0]),
        np.array([
            [ 0.0, -0.5 ],
            [ 0.0, 0.0 ],
            [ 0.5, 0.0 ],
            [-0.5, 0.5 ]
        ])
    ),
    # Center (ξ=0, η=0)
    (
        np.array([0.0,  0.0]),
        np.array([
            [-0.25, -0.25],
            [ 0.25, -0.25],
            [ 0.25,  0.25],
            [-0.25,  0.25]
        ])
    ),
    # Intermediate point (ξ=0.5, η=0.5)
    (
        np.array([0.5, 0.5]),
        np.array([
            [-0.25*(1-0.5), -0.25*(1-0.5)],  # dN1 => (-0.25*0.5, -0.25*0.5) => (-0.125, -0.125)
            [ 0.25*(1-0.5), -0.25*(1+0.5)],  # dN2 => (0.125, -0.25*1.5= -0.375)
            [ 0.25*(1+0.5),  0.25*(1+0.5)],  # dN3 => (0.25*1.5=0.375, 0.375)
            [-0.25*(1+0.5),  0.25*(1-0.5)],  # dN4 => (-0.25*1.5= -0.375, 0.25*0.5=0.125)
        ])
    ),
])
def test_D2_nn4_quad_dxi(xi_vec, expected):
    """
    Test the D2_nn4_quad_dxi function at various (ξ,η) points, ensuring:
      - The returned array has shape (4,2).
      - The derivative values match the standard bilinear formula (within tolerance).
    """
    result = di.D2_nn4_quad_dxi(xi_vec)

    # Check shape
    assert result.shape == (4, 2), f"Expected shape (4,2) but got {result.shape}."

    # Compare numeric values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi_vec}, expected {expected.ravel()} but got {result.ravel()}."
    )


@pytest.mark.parametrize("xi, expected", [
    # (ξ, η) = (-1,-1) => N1=1, others 0
    (np.array([-1.0, -1.0]), np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])),
    # (ξ, η) = (1,-1) => N2=1, others 0
    (np.array([1.0, -1.0]), np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])),
    # (ξ, η) = (1,1) => N3=1, others 0
    (np.array([1.0, 1.0]), np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])),
    # (ξ, η) = (-1,1) => N4=1, others 0
    (np.array([-1.0, 1.0]), np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])),
    # (ξ, η) = (0,0) => centroid (updated expected values based on function output)
    (np.array([0.0, 0.0]), np.array([[-0.25], [-0.25], [-0.25], [-0.25], [0.5], [0.5], [0.5], [0.5]])),
    # Midpoints of edges
    (np.array([0.0, -1.0]), np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])),
    (np.array([1.0, 0.0]), np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])),
    (np.array([0.0, 1.0]), np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0]])),
    (np.array([-1.0, 0.0]), np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])),
])
def test_D2_nn8_quad(xi, expected):
    """
    Test the D2_nn8_quad function, verifying:
      - The returned array has shape (8,1)
      - The shape function values match expected results (within tolerance).
    """
    result = di.D2_nn8_quad(xi)
    # Check shape
    assert result.shape == (8, 1), f"Expected shape (8,1) but got {result.shape}."
    # Check values
    assert np.allclose(result, expected, atol=1e-12), (
        f"For xi={xi}, expected {expected.ravel()} but got {result.ravel()}."
    )


def reference_d2_nn8_quad_dxi(xi, eta):
    """
    Compute the derivatives (dN_dxi) for the 8-node quadratic quadrilateral element
    at a given (xi, eta), strictly following the formula in the docstring.
    Returns an (8,2) array.
    """
    dN = np.zeros((8, 2))
    # Corner Nodes
    # dN1/dξ =  0.25 * (1 - η) * (2ξ + η)      , dN1/dη =  0.25 * (1 - ξ) * (ξ + 2η)
    dN[0, 0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
    dN[0, 1] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)

    # dN2/dξ =  0.25 * (1 - η) * (2ξ - η)      , dN2/dη =  0.25 * (1 + ξ) * (2η - ξ)
    dN[1, 0] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
    dN[1, 1] = 0.25 * (1.0 + xi) * (2.0 * eta - xi)

    # dN3/dξ =  0.25 * (1 + η) * (2ξ + η)      , dN3/dη =  0.25 * (1 + ξ) * (2η + ξ)
    dN[2, 0] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
    dN[2, 1] = 0.25 * (1.0 + xi) * (2.0 * eta + xi)

    # dN4/dξ =  0.25 * (1 + η) * (2ξ - η)      , dN4/dη =  0.25 * (1 - ξ) * (2η - ξ)
    dN[3, 0] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
    dN[3, 1] = 0.25 * (1.0 - xi) * (2.0 * eta - xi)

    # Mid-Side Nodes
    # dN5/dξ = -ξ * (1 - η)        , dN5/dη = -0.5 * (1 - ξ²)
    dN[4, 0] = -xi * (1.0 - eta)
    dN[4, 1] = -0.5 * (1.0 - xi * xi)

    # dN6/dξ = 0.5 * (1 - η²)      , dN6/dη = -(1 + ξ) * η
    dN[5, 0] = 0.5 * (1.0 - eta * eta)
    dN[5, 1] = -(1.0 + xi) * eta

    # dN7/dξ = -ξ * (1 + η)        , dN7/dη =  0.5 * (1 - ξ²)
    dN[6, 0] = -xi * (1.0 + eta)
    dN[6, 1] = 0.5 * (1.0 - xi * xi)

    # dN8/dξ = -0.5 * (1 - η²)     , dN8/dη = -(1 - ξ) * η
    dN[7, 0] = -0.5 * (1.0 - eta * eta)
    dN[7, 1] = -(1.0 - xi) * eta

    return dN


@pytest.mark.parametrize("xi_vec", [
    # Corners
    np.array([-1.0, -1.0]),
    np.array([ 1.0, -1.0]),
    np.array([ 1.0,  1.0]),
    np.array([-1.0,  1.0]),
    # Edges
    np.array([-1.0, 0.0]),
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([0.0, -1.0]),
    # Center
    np.array([0.0, 0.0]),
    # Some intermediate points
    np.array([0.5, -0.5]),
    np.array([-0.3, 0.7]),
])
def test_D2_nn8_quad_dxi(xi_vec):
    """
    Test D2_nn8_quad_dxi at various (xi, η) points, ensuring:
      - The returned array has shape (8,2).
      - The derivative values match the formula from the docstring.
    """
    # Evaluate the function under test
    dN_num = di.D2_nn8_quad_dxi(xi_vec)
    # Evaluate a reference version
    xi, eta = xi_vec
    dN_ref = reference_d2_nn8_quad_dxi(xi, eta)

    # Check shape
    assert dN_num.shape == (8, 2), f"Expected shape (8,2), got {dN_num.shape}."

    # Compare numeric values with a reasonable tolerance
    assert np.allclose(dN_num, dN_ref, atol=1e-12), (
        f"For xi={xi_vec}, difference:\n{dN_num - dN_ref}"
    )


###########################################################
#  TEST GAUSSIAN INTEGRATION POINTS AND WEIGHTS
###########################################################
def integrate_function_gauss_1d(f, num_pts):
    """
    Numerically integrates a given function f over the interval [-1,1]
    using Gauss-Legendre quadrature.
    """
    xi = di.gauss_points_1d(num_pts).flatten()
    w = di.gauss_weights_1d(num_pts).flatten()
    integral = np.sum(w * f(xi))
    return integral


@pytest.mark.parametrize("num_pts, expected", [
    # Integrating f(x) = 1 over [-1,1] should return 2 (exact result)
    (1, 2.0),
    (2, 2.0),
    (3, 2.0),
    # Integrating f(x) = x over [-1,1] should return 0 (due to symmetry)
    (1, 0.0),
    (2, 0.0),
    (3, 0.0),
    # Integrating f(x) = x^2 over [-1,1] should return 2/3 (exact result), valid for 2-point and 3-point rules
    (2, 0.6666666666666666),
    (3, 0.6666666666666666),
])
def test_gauss_quadrature_1d(num_pts, expected):
    """
    Test the combined Gauss points and weights for integration accuracy
    over different polynomial functions.
    """
    functions = {
        2.0: lambda x: np.ones_like(x),
        0.0: lambda x: x,
        0.6666666666666666: lambda x: x**2
    }
    result = integrate_function_gauss_1d(functions[expected], num_pts)
    assert np.allclose(result, expected, atol=1e-12), (
        f"For num_pts={num_pts}, expected integral {expected} but got {result}."
    )


def integrate_function_gauss_2d(f, num_pts):
    """
    Numerically integrates a given function f over the reference triangle
    using Gauss-Legendre quadrature.
    """
    xi = di.gauss_points_2d_triangle(num_pts)
    w = di.gauss_weights_2d_triangle(num_pts).flatten()
    integral = np.sum(w * f(xi[0], xi[1]))
    return integral


@pytest.mark.parametrize("num_pts, expected", [
    # Integrating f(x, y) = 1 over reference triangle (area = 0.5)
    (1, 0.5),
    (3, 0.5),
    (4, 0.5),
    # Integrating f(x, y) = x over reference triangle (exact integral = 1/6)
    (3, 1/6),
    (4, 1/6),
    # Integrating f(x, y) = x^2 over reference triangle (exact integral = 1/12), valid only for 4-point rule
    (4, 1/12),
])
def test_gauss_quadrature_2d(num_pts, expected):
    """
    Test the combined Gauss points and weights for integration accuracy
    over different polynomial functions in 2D.
    """
    functions = {
        0.5: lambda x, y: np.ones_like(x),
        1/6: lambda x, y: x,
        1/12: lambda x, y: x**2
    }
    result = integrate_function_gauss_2d(functions[expected], num_pts)
    assert np.allclose(result, expected, atol=1e-12), (
        f"For num_pts={num_pts}, expected integral {expected} but got {result}."
    )


def integrate_function_gauss_2d_quad(f, num_pts):
    """
    Numerically integrates a given function f over the reference quadrilateral
    using Gauss-Legendre quadrature.
    """
    xi = di.gauss_points_2d_quad(num_pts)
    w = di.gauss_weights_2d_quad(num_pts).flatten()
    integral = np.sum(w * f(xi[0], xi[1]))
    return integral


@pytest.mark.parametrize("num_pts, expected", [
    # Integrating f(x, y) = 1 over reference square (area = 4.0)
    (1, 4.0),
    (4, 4.0),
    (9, 4.0),
    # Integrating f(x, y) = x over reference square (exact integral = 0, symmetry)
    (4, 0.0),
    (9, 0.0),
    # Integrating f(x, y) = x^2 over reference square (exact integral = 4/3)
    (4, 4/3),
    (9, 4/3),
    # Integrating f(x, y) = x^3 over reference square (exact integral = 0, symmetry)
    (9, 0.0),
    # Integrating f(x, y) = x^4 over reference square (exact integral = 4/5)
    (9, 4/5),
    # Integrating f(x, y) = x^5 over reference square (exact integral = 0, symmetry)
    (9, 0.0),
])
def test_gauss_quadrature_2d_quad(num_pts, expected):
    """
    Test the combined Gauss points and weights for integration accuracy
    over different polynomial functions in 2D quadrilateral elements.
    """
    functions = {
        4.0: lambda x, y: np.ones_like(x),
        0.0: lambda x, y: x,
        4/3: lambda x, y: x**2,
        4/5: lambda x, y: x**4,
        4/7: lambda x, y: x**6,
    }
    result = integrate_function_gauss_2d_quad(functions[expected], num_pts)
    assert np.allclose(result, expected, atol=1e-12), (
        f"For num_pts={num_pts}, expected integral {expected} but got {result}."
    )


###########################################################
#  PARTITION OF UNITY CHECKS
###########################################################
@pytest.mark.parametrize("num_pts, shape_function", [
    (1, di.D1_nn3),
    (2, di.D1_nn3),
    (3, di.D1_nn3),
    (1, di.D1_nn2),
    (2, di.D1_nn2),
    (3, di.D1_nn2),
])
def test_partition_of_unity_1d(num_pts, shape_function):
    """
    Verify that the shape functions sum to 1 at every Gauss quadrature point for 1D elements
    with either 2-node (D1_nn2) or 3-node (D1_nn3) elements.
    """
    xi = di.gauss_points_1d(num_pts)
    for i in range(xi.shape[1]):  # Iterate over Gauss points
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        assert np.allclose(np.sum(N), 1.0, atol=1e-12), f"Partition of unity failed for {shape_function.__name__} at Gauss point {xi[:, i]}"


@pytest.mark.parametrize("num_pts, shape_function", [
    (1, di.D2_nn3_tri),
    (3, di.D2_nn3_tri),
    (4, di.D2_nn3_tri),
    (1, di.D2_nn6_tri),
    (3, di.D2_nn6_tri),
    (4, di.D2_nn6_tri),
])
def test_partition_of_unity_2d_tri(num_pts, shape_function):
    """
    Verify that the shape functions sum to 1 at every Gauss quadrature point for 2D triangular elements
    with either 3-node (D2_nn3_tri) or 6-node (D2_nn6_tri) elements.
    """
    xi = di.gauss_points_2d_triangle(num_pts)
    for i in range(xi.shape[1]):  # Iterate over Gauss points
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        assert np.allclose(np.sum(N), 1.0, atol=1e-12), f"Partition of unity failed for {shape_function.__name__} at Gauss point {xi[:, i]}"


@pytest.mark.parametrize("num_pts, shape_function", [
    (1, di.D2_nn4_quad),
    (4, di.D2_nn4_quad),
    (9, di.D2_nn4_quad),
    (1, di.D2_nn8_quad),
    (4, di.D2_nn8_quad),
    (9, di.D2_nn8_quad),
])
def test_partition_of_unity_2d_quad(num_pts, shape_function):
    """
    Verify that the shape functions sum to 1 at every Gauss quadrature point for 2D quadrilateral elements
    with either 4-node (D2_nn4_quad) or 8-node (D2_nn8_quad) elements.
    """
    xi = di.gauss_points_2d_quad(num_pts)
    for i in range(xi.shape[1]):  # Iterate over Gauss points
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        assert np.allclose(np.sum(N), 1.0, atol=1e-12), f"Partition of unity failed for {shape_function.__name__} at Gauss point {xi[:, i]}"


###########################################################
#  COMPLETENESS CHECKS
###########################################################
@pytest.mark.parametrize("num_pts, shape_function, nodal_coords", [
    (1, di.D1_nn3, np.array([-1.0, 1.0, 0.0])),
    (2, di.D1_nn3, np.array([-1.0, 1.0, 0.0])),
    (3, di.D1_nn3, np.array([-1.0, 1.0, 0.0])),
    (1, di.D1_nn2, np.array([-1.0, 1.0])),
    (2, di.D1_nn2, np.array([-1.0, 1.0])),
    (3, di.D1_nn2, np.array([-1.0, 1.0])),
])
def test_completeness_1d(num_pts, shape_function, nodal_coords):
    """
    Verify that shape functions reproduce linear functions exactly for 1D elements.
    """
    xi = di.gauss_points_1d(num_pts)
    
    for i in range(xi.shape[1]):  # Iterate over Gauss points
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        x_interp = np.sum(N.flatten() * nodal_coords)  # Linear interpolation
        expected_value = xi[0, i]  # Linear function being tested
        
        assert np.allclose(x_interp, expected_value, atol=1e-10), (
            f"Completeness failed for {shape_function.__name__} at Gauss point {xi[:, i]}: Expected {expected_value}, Got {x_interp}"
        )


@pytest.mark.parametrize("num_pts, shape_function, nodal_coords", [
    (1, di.D2_nn3_tri, np.array([[1, 0], [0, 1], [0, 0]])),
    (3, di.D2_nn3_tri, np.array([[1, 0], [0, 1], [0, 0]])),
    (4, di.D2_nn3_tri, np.array([[1, 0], [0, 1], [0, 0]])),
    (1, di.D2_nn6_tri, np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])),
    (3, di.D2_nn6_tri, np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])),
    (4, di.D2_nn6_tri, np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])),
])
def test_completeness_2d_tri(num_pts, shape_function, nodal_coords):
    """
    Verify that shape functions reproduce linear functions exactly for 2D triangular elements.
    """
    xi = di.gauss_points_2d_triangle(num_pts)
    
    for i in range(xi.shape[1]):  # Iterate over Gauss points
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        x_interp = np.sum(N.flatten() * nodal_coords[:, 0])  # Linear interpolation in x
        y_interp = np.sum(N.flatten() * nodal_coords[:, 1])  # Linear interpolation in y
        
        assert np.allclose(x_interp, xi[0, i], atol=1e-10), (
            f"Completeness failed at Gauss point {xi[:, i]} in x-direction: Expected {xi[0, i]}, Got {x_interp}"
        )
        assert np.allclose(y_interp, xi[1, i], atol=1e-10), (
            f"Completeness failed at Gauss point {xi[:, i]} in y-direction: Expected {xi[1, i]}, Got {y_interp}"
        )


@pytest.mark.parametrize("num_pts, shape_function, nodal_coords", [
    (1, di.D2_nn4_quad, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])),
    (4, di.D2_nn4_quad, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])),
    (9, di.D2_nn4_quad, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])),
    (1, di.D2_nn8_quad, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])),
    (4, di.D2_nn8_quad, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])),
    (9, di.D2_nn8_quad, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])),
])
def test_completeness_2d_quad(num_pts, shape_function, nodal_coords):
    """
    Verify that shape functions reproduce linear functions exactly for 2D quadrilateral elements.
    """
    xi = di.gauss_points_2d_quad(num_pts)
    
    for i in range(xi.shape[1]):  # Iterate over Gauss points
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        x_interp = np.sum(N.flatten() * nodal_coords[:, 0])  # Linear interpolation in x
        y_interp = np.sum(N.flatten() * nodal_coords[:, 1])  # Linear interpolation in y
        
        assert np.allclose(x_interp, xi[0, i], atol=1e-10), (
            f"Completeness failed at Gauss point {xi[:, i]} in x-direction: Expected {xi[0, i]}, Got {x_interp}"
        )
        assert np.allclose(y_interp, xi[1, i], atol=1e-10), (
            f"Completeness failed at Gauss point {xi[:, i]} in y-direction: Expected {xi[1, i]}, Got {y_interp}"
        )


###########################################################
#  INTEGRATION OF SHAPE FUNCTIONS CHECKS
###########################################################
def integrate_shape_function_1d(shape_function, num_pts, expected_integral, node_order):
    """
    Numerically integrate each shape function over the 1D element using Gauss quadrature,
    correctly ordering shape function contributions.
    """
    xi = di.gauss_points_1d(num_pts).flatten()  # Ensure (num_pts,)
    w = di.gauss_weights_1d(num_pts).flatten()
    num_nodes = expected_integral.shape[0]
    integral = np.zeros(num_nodes)
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        N = shape_function(np.array([xi[i]]))  # Pass xi[i] as a 1D array
        integral += N.flatten()[node_order] * w[i]  # Reorder contributions based on node order
    
    assert np.allclose(integral, expected_integral, atol=1e-10), (
        f"Shape function integration failed for {shape_function.__name__}: Expected {expected_integral}, Got {integral}"
    )

    assert np.isclose(np.sum(integral), 2.0, atol=1e-10)


@pytest.mark.parametrize("num_pts, shape_function, expected_integral, node_order", [
    (1, di.D1_nn2, np.array([1, 1]), [0, 1]),  # Left, Right ordering for 2-node element
    (2, di.D1_nn2, np.array([1, 1]), [0, 1]),
    (3, di.D1_nn2, np.array([1, 1]), [0, 1]),
    (2, di.D1_nn3, np.array([1/3, 4/3, 1/3]), [0, 2, 1]),  # Left, Center, Right ordering for 3-node element
    (3, di.D1_nn3, np.array([1/3, 4/3, 1/3]), [0, 2, 1]),
])
def test_shape_function_integration_1d(num_pts, shape_function, expected_integral, node_order):
    """
    Test the numerical integration of shape functions over the 1D reference element,
    correctly handling node ordering.
    """
    integrate_shape_function_1d(shape_function, num_pts, expected_integral, node_order)


def integrate_shape_function_gradient_1d(shape_function_dxi, num_pts, node_order):
    """
    Numerically integrate the gradient of shape functions over the 1D element using Gauss quadrature,
    correctly ordering shape function derivative contributions.
    """
    xi = di.gauss_points_1d(num_pts).flatten()  # Ensure (num_pts,)
    w = di.gauss_weights_1d(num_pts).flatten()
    num_nodes = len(node_order)
    integral = np.zeros(num_nodes)
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        dN_dxi = shape_function_dxi(np.array([xi[i]]))  # Compute shape function derivatives at Gauss point
        integral += dN_dxi.flatten()[node_order] * w[i]  # Reorder contributions based on node order
    
    assert np.isclose(np.sum(integral), 0.0, atol=1e-10), (
        f"Gradient integral sum failed for {shape_function_dxi.__name__}: Expected 0.0, Got {np.sum(integral)}"
    )


@pytest.mark.parametrize("num_pts, shape_function_dxi, node_order", [
    (2, di.D1_nn2_dxi, [0, 1]),  # Integral of dN/dxi over [-1,1] should sum to 0 for linear elements
    (3, di.D1_nn2_dxi, [0, 1]),
    (2, di.D1_nn3_dxi, [0, 2, 1]),  # Integral of dN/dxi over [-1,1] should sum to 0 for quadratic elements
    (3, di.D1_nn3_dxi, [0, 2, 1]),
])
def test_shape_function_gradient_integration_1d(num_pts, shape_function_dxi, node_order):
    """
    Test the numerical integration of shape function gradients over the 1D reference element,
    ensuring their total integral sums to zero as expected.
    """
    integrate_shape_function_gradient_1d(shape_function_dxi, num_pts, node_order)


def integrate_shape_function_derivative_product_1d(shape_function_dxi, num_pts, node_order):
    """
    Numerically integrate the product of shape function derivatives over the 1D element
    using Gauss quadrature to verify symmetry.
    """
    xi = di.gauss_points_1d(num_pts).flatten()  # Ensure (num_pts,)
    w = di.gauss_weights_1d(num_pts).flatten()
    num_nodes = len(node_order)
    integral_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        dN_dxi = shape_function_dxi(np.array([xi[i]])).flatten()[node_order]  # Compute shape function derivatives at Gauss point
        integral_matrix += np.outer(dN_dxi, dN_dxi) * w[i]  # Compute weighted outer product
    
    assert np.allclose(integral_matrix, integral_matrix.T, atol=1e-10), "Shape function derivative symmetry failed."


@pytest.mark.parametrize("num_pts, shape_function_dxi, node_order", [
    (2, di.D1_nn2_dxi, [0, 1]),  # Symmetry check for linear elements
    (3, di.D1_nn2_dxi, [0, 1]),
    (2, di.D1_nn3_dxi, [0, 2, 1]),  # Symmetry check for quadratic elements
    (3, di.D1_nn3_dxi, [0, 2, 1]),
])
def test_shape_function_derivative_symmetry_1d(num_pts, shape_function_dxi, node_order):
    """
    Test the symmetry of the shape function derivative integral over the 1D reference element.
    """
    integrate_shape_function_derivative_product_1d(shape_function_dxi, num_pts, node_order)


def integrate_shape_function_2d_tri(shape_function, num_pts, node_order):
    """
    Numerically integrate each shape function over the 2D triangular element using Gauss quadrature,
    correctly ordering shape function contributions.
    """
    xi = di.gauss_points_2d_triangle(num_pts)  # Gauss points (2, num_pts)
    w = di.gauss_weights_2d_triangle(num_pts).flatten()  # Gauss weights (num_pts,)
    num_nodes = len(node_order)
    integral = np.zeros(num_nodes)
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        integral += N.flatten()[node_order] * w[i]  # Reorder contributions based on node order
    
    assert np.isclose(np.sum(integral), 0.5, atol=1e-9), "Total integral does not match expected reference area."


@pytest.mark.parametrize("num_pts, shape_function, node_order", [
    (1, di.D2_nn3_tri, [0, 1, 2]),  # Linear 3-node element
    (3, di.D2_nn3_tri, [0, 1, 2]),
    (4, di.D2_nn3_tri, [0, 1, 2]),
    (1, di.D2_nn6_tri, [0, 1, 2, 3, 4, 5]),  # Quadratic 6-node element
    (3, di.D2_nn6_tri, [0, 1, 2, 3, 4, 5]),
    (4, di.D2_nn6_tri, [0, 1, 2, 3, 4, 5]),
])
def test_shape_function_integration_2d_tri(num_pts, shape_function, node_order):
    """
    Test the numerical integration of shape functions over the 2D triangular reference element,
    ensuring that the total integral matches the expected area.
    """
    integrate_shape_function_2d_tri(shape_function, num_pts, node_order)


def integrate_shape_function_gradient_2d_tri(shape_function_dxi, num_pts, node_order):
    """
    Numerically integrate the gradient of shape functions over the 2D triangular element using Gauss quadrature,
    ensuring that the total integral of the gradient sums to zero as expected.
    """
    xi = di.gauss_points_2d_triangle(num_pts)  # Gauss points (2, num_pts)
    w = di.gauss_weights_2d_triangle(num_pts).flatten()  # Gauss weights (num_pts,)
    num_nodes = len(node_order)
    integral = np.zeros((num_nodes, 2))  # Two components for 2D gradients (dN/dξ, dN/dη)

    for i in range(num_pts):  # Iterate over Gauss points correctly
        dN_dxi = shape_function_dxi(xi[:, i])  # Compute shape function gradients at Gauss point
        integral += dN_dxi[node_order, :] * w[i]  # Reorder contributions based on node order

    assert np.allclose(np.sum(integral, axis=0), [0.0, 0.0], atol=1e-10), "Gradient integral sum does not match expected zero vector."


def integrate_shape_function_2d_quad(shape_function, num_pts, node_order):
    """
    Numerically integrate each shape function over the 2D quadrilateral element using Gauss quadrature,
    ensuring that the total integral matches the reference element area (4.0).
    """
    xi = di.gauss_points_2d_quad(num_pts)  # Gauss points (2, num_pts)
    w = di.gauss_weights_2d_quad(num_pts).flatten()  # Gauss weights (num_pts,)
    num_nodes = len(node_order)
    integral = np.zeros(num_nodes)

    for i in range(num_pts):  # Iterate over Gauss points correctly
        N = shape_function(xi[:, i])  # Compute shape functions at Gauss point
        integral += N.flatten()[node_order] * w[i]  # Reorder contributions based on node order

    assert np.isclose(np.sum(integral), 4.0, atol=1e-9), "Total integral does not match expected reference area."


@pytest.mark.parametrize("num_pts, shape_function, node_order", [
    (1, di.D2_nn4_quad, [0, 1, 2, 3]),  # Bilinear 4-node element
    (4, di.D2_nn4_quad, [0, 1, 2, 3]),
    (9, di.D2_nn4_quad, [0, 1, 2, 3]),
    (1, di.D2_nn8_quad, [0, 1, 2, 3, 4, 5, 6, 7]),  # Quadratic 8-node element
    (4, di.D2_nn8_quad, [0, 1, 2, 3, 4, 5, 6, 7]),
    (9, di.D2_nn8_quad, [0, 1, 2, 3, 4, 5, 6, 7]),
])
def test_shape_function_integration_2d_quad(num_pts, shape_function, node_order):
    """
    Test the numerical integration of shape functions over the 2D quadrilateral reference element,
    ensuring that the total integral matches the expected area.
    """
    integrate_shape_function_2d_quad(shape_function, num_pts, node_order)


@pytest.mark.parametrize("num_pts, shape_function_dxi, node_order", [
    (1, di.D2_nn3_tri_dxi, [0, 1, 2]),  # Linear 3-node element
    (3, di.D2_nn3_tri_dxi, [0, 1, 2]),
    (4, di.D2_nn3_tri_dxi, [0, 1, 2]),
    (1, di.D2_nn6_tri_dxi, [0, 1, 2, 3, 4, 5]),  # Quadratic 6-node element
    (3, di.D2_nn6_tri_dxi, [0, 1, 2, 3, 4, 5]),
    (4, di.D2_nn6_tri_dxi, [0, 1, 2, 3, 4, 5]),
])
def test_shape_function_gradient_integration_2d_tri(num_pts, shape_function_dxi, node_order):
    """
    Test the numerical integration of shape function gradients over the 2D triangular reference element,
    ensuring that the total integral of gradients sums to zero.
    """
    integrate_shape_function_gradient_2d_tri(shape_function_dxi, num_pts, node_order)


def integrate_shape_function_derivative_product_2d_tri(shape_function_dxi, num_pts, node_order):
    """
    Numerically integrate the product of shape function derivatives over the 2D triangular element
    using Gauss quadrature to verify symmetry.
    """
    xi = di.gauss_points_2d_triangle(num_pts)  # Gauss points (2, num_pts)
    w = di.gauss_weights_2d_triangle(num_pts).flatten()  # Gauss weights (num_pts,)
    num_nodes = len(node_order)
    integral_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        dN_dxi = shape_function_dxi(xi[:, i])  # Compute shape function derivatives at Gauss point
        dN_dxi = dN_dxi[node_order, :]  # Reorder contributions based on node order
        integral_matrix += (dN_dxi @ dN_dxi.T) * w[i]  # Compute weighted matrix product
    
    assert np.allclose(integral_matrix, integral_matrix.T, atol=1e-10), "Shape function derivative symmetry failed."


@pytest.mark.parametrize("num_pts, shape_function_dxi, node_order", [
    (1, di.D2_nn3_tri_dxi, [0, 1, 2]),  # Symmetry check for linear 3-node element
    (3, di.D2_nn3_tri_dxi, [0, 1, 2]),
    (4, di.D2_nn3_tri_dxi, [0, 1, 2]),
    (1, di.D2_nn6_tri_dxi, [0, 1, 2, 3, 4, 5]),  # Symmetry check for quadratic 6-node element
    (3, di.D2_nn6_tri_dxi, [0, 1, 2, 3, 4, 5]),
    (4, di.D2_nn6_tri_dxi, [0, 1, 2, 3, 4, 5]),
])
def test_shape_function_derivative_symmetry_2d_tri(num_pts, shape_function_dxi, node_order):
    """
    Test the symmetry of the shape function derivative integral over the 2D triangular reference element.
    """
    integrate_shape_function_derivative_product_2d_tri(shape_function_dxi, num_pts, node_order)


def integrate_shape_function_gradient_2d_quad(shape_function_dxi, num_pts, node_order):
    """
    Numerically integrate the gradient of shape functions over the 2D quadrilateral element using Gauss quadrature,
    ensuring that the total integral of the gradient sums to zero as expected.
    """
    xi = di.gauss_points_2d_quad(num_pts)  # Gauss points (2, num_pts)
    w = di.gauss_weights_2d_quad(num_pts).flatten()  # Gauss weights (num_pts,)
    num_nodes = len(node_order)
    integral = np.zeros((num_nodes, 2))  # Two components for 2D gradients (dN/dξ, dN/dη)
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        dN_dxi = shape_function_dxi(xi[:, i])  # Compute shape function gradients at Gauss point
        integral += dN_dxi[node_order, :] * w[i]  # Reorder contributions based on node order
    
    assert np.allclose(np.sum(integral, axis=0), [0.0, 0.0], atol=1e-10), "Gradient integral sum does not match expected zero vector."


@pytest.mark.parametrize("num_pts, shape_function_dxi, node_order", [
    (1, di.D2_nn4_quad_dxi, [0, 1, 2, 3]),  # Bilinear 4-node element
    (4, di.D2_nn4_quad_dxi, [0, 1, 2, 3]),
    (9, di.D2_nn4_quad_dxi, [0, 1, 2, 3]),
    (1, di.D2_nn8_quad_dxi, [0, 1, 2, 3, 4, 5, 6, 7]),  # Quadratic 8-node element
    (4, di.D2_nn8_quad_dxi, [0, 1, 2, 3, 4, 5, 6, 7]),
    (9, di.D2_nn8_quad_dxi, [0, 1, 2, 3, 4, 5, 6, 7]),
])
def test_shape_function_gradient_integration_2d_quad(num_pts, shape_function_dxi, node_order):
    """
    Test the numerical integration of shape function gradients over the 2D quadrilateral reference element,
    ensuring that the total integral of gradients sums to zero.
    """
    integrate_shape_function_gradient_2d_quad(shape_function_dxi, num_pts, node_order)


def integrate_shape_function_derivative_product_2d_quad(shape_function_dxi, num_pts, node_order):
    """
    Numerically integrate the product of shape function derivatives over the 2D quadrilateral element
    using Gauss quadrature to verify symmetry.
    """
    xi = di.gauss_points_2d_quad(num_pts)  # Gauss points (2, num_pts)
    w = di.gauss_weights_2d_quad(num_pts).flatten()  # Gauss weights (num_pts,)
    num_nodes = len(node_order)
    integral_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_pts):  # Iterate over Gauss points correctly
        dN_dxi = shape_function_dxi(xi[:, i])  # Compute shape function derivatives at Gauss point
        dN_dxi = dN_dxi[node_order, :]  # Reorder contributions based on node order
        integral_matrix += (dN_dxi @ dN_dxi.T) * w[i]  # Compute weighted matrix product
    
    assert np.allclose(integral_matrix, integral_matrix.T, atol=1e-10), "Shape function derivative symmetry failed."


@pytest.mark.parametrize("num_pts, shape_function_dxi, node_order", [
    (1, di.D2_nn4_quad_dxi, [0, 1, 2, 3]),  # Symmetry check for bilinear 4-node element
    (4, di.D2_nn4_quad_dxi, [0, 1, 2, 3]),
    (9, di.D2_nn4_quad_dxi, [0, 1, 2, 3]),
    (1, di.D2_nn8_quad_dxi, [0, 1, 2, 3, 4, 5, 6, 7]),  # Symmetry check for quadratic 8-node element
    (4, di.D2_nn8_quad_dxi, [0, 1, 2, 3, 4, 5, 6, 7]),
    (9, di.D2_nn8_quad_dxi, [0, 1, 2, 3, 4, 5, 6, 7]),
])
def test_shape_function_derivative_symmetry_2d_quad(num_pts, shape_function_dxi, node_order):
    """
    Test the symmetry of the shape function derivative integral over the 2D quadrilateral reference element.
    """
    integrate_shape_function_derivative_product_2d_quad(shape_function_dxi, num_pts, node_order)


###########################################################
#  TEST FACE NODES
###########################################################
@pytest.mark.parametrize("func, face, expected", [
    # 1D Elements
    (di.get_face_nodes_D1_nn2, 0, [0]),
    (di.get_face_nodes_D1_nn2, 1, [1]),
    (di.get_face_nodes_D1_nn3, 0, [0]),
    (di.get_face_nodes_D1_nn3, 1, [1]),
    # 2D Triangular Elements
    (di.get_face_nodes_D2_nn3_tri, 0, [0, 1]),
    (di.get_face_nodes_D2_nn3_tri, 1, [1, 2]),
    (di.get_face_nodes_D2_nn3_tri, 2, [2, 0]),
    (di.get_face_nodes_D2_nn6_tri, 0, [0, 1, 3]),
    (di.get_face_nodes_D2_nn6_tri, 1, [1, 2, 4]),
    (di.get_face_nodes_D2_nn6_tri, 2, [2, 0, 5]),
    # 2D Quadrilateral Elements
    (di.get_face_nodes_D2_nn4_quad, 0, [0, 1]),
    (di.get_face_nodes_D2_nn4_quad, 1, [1, 2]),
    (di.get_face_nodes_D2_nn4_quad, 2, [2, 3]),
    (di.get_face_nodes_D2_nn4_quad, 3, [3, 0]),
    (di.get_face_nodes_D2_nn8_quad, 0, [0, 1, 4]),
    (di.get_face_nodes_D2_nn8_quad, 1, [1, 2, 5]),
    (di.get_face_nodes_D2_nn8_quad, 2, [2, 3, 6]),
    (di.get_face_nodes_D2_nn8_quad, 3, [3, 0, 7]),
])
def test_get_face_nodes(func, face, expected):
    """
    Test that the correct face nodes are returned for each element type and face index.
    """
    assert func(face) == expected


@pytest.mark.parametrize("func, invalid_face", [
    (di.get_face_nodes_D1_nn2, 2),
    (di.get_face_nodes_D1_nn3, 3),
    (di.get_face_nodes_D2_nn3_tri, 3),
    (di.get_face_nodes_D2_nn6_tri, 3),
    (di.get_face_nodes_D2_nn4_quad, 4),
    (di.get_face_nodes_D2_nn8_quad, 4),
])
def test_get_face_nodes_invalid(func, invalid_face):
    """
    Test that requesting an invalid face index raises a ValueError.
    """
    with pytest.raises(ValueError):
        func(invalid_face)

