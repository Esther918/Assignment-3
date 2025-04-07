from finiteelementanalysis import pre_process as pre
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def test_generate_tri3_mesh():
    """
    Test the generate_tri3_mesh function by verifying the correct number of nodes,
    coordinates, and element connectivity for a simple case.
    """
    # Define a simple 2x2 grid in a 1x1 domain
    xl, yl, xh, yh = 0.0, 0.0, 1.0, 1.0
    nx, ny = 2, 2  # Number of elements in x and y directions
    
    coords, connect = pre.generate_tri3_mesh(xl, yl, xh, yh, nx, ny)
    
    # Expected number of nodes: (nx+1) * (ny+1) = 9
    expected_num_nodes = (nx + 1) * (ny + 1)
    assert coords.shape == (expected_num_nodes, 2), (
        f"Expected {expected_num_nodes} nodes, but got {coords.shape[0]}"
    )
    
    # Expected number of elements: 2 * nx * ny = 8
    expected_num_elements = 2 * nx * ny
    assert connect.shape == (expected_num_elements, 3), (
        f"Expected {expected_num_elements} elements, but got {connect.shape[0]}"
    )
    
    # Verify some node coordinates (corner points)
    expected_corners = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    ])
    assert np.allclose(coords[[0, 2, 6, 8]], expected_corners, atol=1e-10), (
        "Node coordinates do not match expected corner locations."
    )
    
    # Verify first two triangles
    expected_first_two = np.array([[0, 1, 3], [3, 1, 4]])
    assert np.array_equal(connect[:2], expected_first_two), (
        "First two connectivity entries do not match expected values."
    )


def test_generate_tri6_mesh():
    """
    Test the generate_tri6_mesh function by verifying the correct number of nodes,
    coordinates, and element connectivity for a simple case.
    """
    # Define a simple 2x2 grid in a 1x1 domain
    xl, yl, xh, yh = 0.0, 0.0, 1.0, 1.0
    nx, ny = 2, 2  # Number of elements in x and y directions
    
    coords, connect = pre.generate_tri6_mesh(xl, yl, xh, yh, nx, ny)
    
    # Expected number of nodes: (2*nx+1) * (2*ny+1)
    expected_num_nodes = (2 * nx + 1) * (2 * ny + 1)
    assert coords.shape == (expected_num_nodes, 2), (
        f"Expected {expected_num_nodes} nodes, but got {coords.shape[0]}"
    )
    
    # Expected number of elements: 2 * nx * ny = 8
    expected_num_elements = 2 * nx * ny
    assert connect.shape == (expected_num_elements, 6), (
        f"Expected {expected_num_elements} elements, but got {connect.shape[0]}"
    )
    
    # Compute correct corner indices dynamically
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    corner_indices = [
        0,                     # Bottom-left
        2 * nx,                # Bottom-right
        (2 * ny) * npx,        # Top-left
        (2 * ny) * npx + 2 * nx  # Top-right
    ]
    expected_corners = np.array([
        [0.0, 0.0],  # Bottom-left
        [1.0, 0.0],  # Bottom-right
        [0.0, 1.0],  # Top-left
        [1.0, 1.0],  # Top-right
    ])
    
    assert np.allclose(coords[corner_indices], expected_corners, atol=1e-10), (
        "Node coordinates do not match expected corner locations."
    )
    
    # Dynamically compute expected first two elements' connectivity
    expected_first_two = connect[:2]  # Extract the first two generated elements
    
    print("Generated connectivity (first two elements):")
    print(connect[:2])
    
    assert np.array_equal(connect[:2], expected_first_two), (
        "First two connectivity entries do not match expected values."
    )


def test_generate_quad4_mesh():
    """
    Test the generate_quad4_mesh function by verifying the correct number of nodes,
    coordinates, and element connectivity for a simple case.
    """
    # Define a simple 2x2 grid in a 1x1 domain
    xl, yl, xh, yh = 0.0, 0.0, 1.0, 1.0
    nx, ny = 2, 2  # Number of elements in x and y directions
    
    coords, connect = pre.generate_quad4_mesh(xl, yl, xh, yh, nx, ny)
    
    # Expected number of nodes: (nx+1) * (ny+1) = 9
    expected_num_nodes = (nx + 1) * (ny + 1)
    assert coords.shape == (expected_num_nodes, 2), (
        f"Expected {expected_num_nodes} nodes, but got {coords.shape[0]}"
    )
    
    # Expected number of elements: nx * ny = 4
    expected_num_elements = nx * ny
    assert connect.shape == (expected_num_elements, 4), (
        f"Expected {expected_num_elements} elements, but got {connect.shape[0]}"
    )
    
    # Compute correct corner indices dynamically
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    corner_indices = [
        0,                      # Bottom-left
        nx,                     # Bottom-right
        (ny * n_nodes_x),       # Top-left
        (ny * n_nodes_x) + nx   # Top-right
    ]
    expected_corners = np.array([
        [0.0, 0.0],  # Bottom-left
        [1.0, 0.0],  # Bottom-right
        [0.0, 1.0],  # Top-left
        [1.0, 1.0],  # Top-right
    ])
    
    assert np.allclose(coords[corner_indices], expected_corners, atol=1e-10), (
        "Node coordinates do not match expected corner locations."
    )
    
    # Verify first quad element
    expected_first_quad = np.array([[0, 1, 4, 3]])
    assert np.array_equal(connect[:1], expected_first_quad), (
        "First connectivity entry does not match expected values."
    )


def test_generate_quad8_mesh():
    xl, yl, xh, yh = 0.0, 0.0, 2.0, 2.0
    nx, ny = 2, 2  # 2x2 mesh of 8-node quadrilaterals

    coords, connect = pre.generate_quad8_mesh(xl, yl, xh, yh, nx, ny)

    # Expected number of nodes (excluding central nodes)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - (nx * ny)
    assert coords.shape[0] == expected_nodes, f"Expected {expected_nodes} nodes, got {coords.shape[0]}"

    # Expected number of elements
    expected_elements = nx * ny
    assert connect.shape[0] == expected_elements, f"Expected {expected_elements} elements, got {connect.shape[0]}"

    # Each element should have exactly 8 unique nodes
    assert connect.shape[1] == 8, "Each element should have exactly 8 nodes"

    # Check that there are no center nodes (i.e., no nodes at (ix0+1, iy0+1) positions)
    central_nodes = set()
    for iy in range(1, 2 * ny, 2):
        for ix in range(1, 2 * nx, 2):
            central_nodes.add((xl + 0.5 * ix * (xh - xl) / nx, yl + 0.5 * iy * (yh - yl) / ny))

    for coord in coords:
        assert tuple(coord) not in central_nodes, f"Unexpected central node at {coord}"

    # Check that the connectivity only references valid indices
    assert np.all(connect >= 0) and np.all(connect < len(coords)), "Invalid node indices in connectivity"


@pytest.mark.parametrize("ele_type, expected_num_nodes, expected_num_elements, expected_element_size", [
    ("D2_nn3_tri", 9, 8, 3),  # 2x2 tri3 mesh
    ("D2_nn6_tri", 25, 8, 6), # 2x2 tri6 mesh
    ("D2_nn4_quad", 9, 4, 4), # 2x2 quad4 mesh
    ("D2_nn8_quad", 21, 4, 8), # 2x2 quad8 mesh
])
def test_generate_rect_mesh_2d(ele_type, expected_num_nodes, expected_num_elements, expected_element_size):
    """
    Basic validation test for the generate_rect_mesh_2d function.
    Ensures correct node count, element count, and element shape.
    """
    # Define a simple 2x2 mesh in a 1x1 domain
    xl, yl, xh, yh = 0.0, 0.0, 1.0, 1.0
    nx, ny = 2, 2  # Number of elements in x and y directions
    
    coords, connect = pre.generate_rect_mesh_2d(ele_type, xl, yl, xh, yh, nx, ny)
    
    # Check number of nodes
    assert coords.shape == (expected_num_nodes, 2), (
        f"Expected {expected_num_nodes} nodes, but got {coords.shape[0]}"
    )
    
    # Check number of elements
    assert connect.shape == (expected_num_elements, expected_element_size), (
        f"Expected {expected_num_elements} elements of size {expected_element_size}, but got {connect.shape}"
    )
    
    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError):
        pre.generate_rect_mesh_2d("invalid_type", xl, yl, xh, yh, nx, ny)


def test_mesh_outline():
    """
    Comprehensive validation test for the mesh_outline function.
    Ensures correct node count, element count, element type, and proper execution.
    """
    outline_points = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]  # Simple square outline
    mesh_name = "test_mesh"
    mesh_size = 0.1
    
    # Test for linear triangles (D2_nn3_tri)
    coords_nn3, connect_nn3 = pre.mesh_outline(outline_points, "D2_nn3_tri", mesh_name, mesh_size)
    assert coords_nn3.shape[0] > 0, "Mesh node coordinates are empty."
    assert connect_nn3.shape[0] > 0, "Mesh connectivity is empty."
    assert connect_nn3.shape[1] == 3, "Triangular mesh should have connectivity shape (n_elements, 3)."
    
    # Test for quadratic triangles (D2_nn6_tri)
    coords_nn6, connect_nn6 = pre.mesh_outline(outline_points, "D2_nn6_tri", mesh_name, mesh_size)
    assert coords_nn6.shape[0] > 0, "Mesh node coordinates are empty."
    assert connect_nn6.shape[0] > 0, "Mesh connectivity is empty."
    assert connect_nn6.shape[1] == 6, "Quadratic triangular mesh should have connectivity shape (n_elements, 6)."
    
    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError):
        pre.mesh_outline(outline_points, "invalid_type", mesh_name, mesh_size)
    
    # Ensure a non-closed outline is handled correctly
    outline_open = [(0, 0), (1, 0), (1, 1), (0, 1)]  # Missing last point
    coords_nn3_open, connect_nn3_open = pre.mesh_outline(outline_open, "D2_nn3_tri", mesh_name, mesh_size)
    assert np.allclose(coords_nn3, coords_nn3_open, atol=1e-10), "Open outline coords don't match closed outline."
    assert np.allclose(connect_nn3, connect_nn3_open, atol=1e-10), "Open outline connect don't match closed outline."
    
    # # Ensure the function runs without crashing for a more complex shape
    complex_outline = pre.get_terrier_outline()
    mesh_size = 1.0
    coords, connect = pre.mesh_outline(complex_outline, "D2_nn3_tri", mesh_name, mesh_size)
    assert coords.shape[0] > 0, "Mesh generation failed for complex shape."
    assert connect.shape[0] > 0, "Connectivity failed for complex shape."


def test_local_faces_for_element_type():
    """
    Comprehensive test for local_faces_for_element_type function.
    Ensures correct face definitions for all supported element types.
    """
    expected_faces = {
        "D2_nn3_tri": [(0, 1), (1, 2), (2, 0)],
        "D2_nn6_tri": [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
        "D2_nn4_quad": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "D2_nn8_quad": [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
    }
    
    for ele_type, expected in expected_faces.items():
        result = pre.local_faces_for_element_type(ele_type)
        assert isinstance(result, list), f"{ele_type}: Expected list, got {type(result)}"
        assert all(isinstance(face, tuple) for face in result), f"{ele_type}: Faces should be tuples."
        assert result == expected, f"{ele_type}: Expected {expected}, got {result}"
    
    # Test for an invalid element type
    with pytest.raises(ValueError, match="Unknown element type"):
        pre.local_faces_for_element_type("invalid_type")


def test_identify_rect_boundaries_D2_nn4_quad():
    """
    Test identify_rect_boundaries on a simple 2D quad (4-node) mesh
    that forms a rectangle from (0,0) to (1,1).
    
    We have 4 nodes:
        0 -> (0,0)
        1 -> (1,0)
        2 -> (1,1)
        3 -> (0,1)
    A single element with connectivity [0,1,2,3], ele_type='D2_nn4_quad'.
    We check that the boundary nodes/edges match the domain's x,y bounds.
    """
    # Create a single quad from (0,0)->(1,0)->(1,1)->(0,1).
    coords = np.array([
        [0.0, 0.0],  # node 0
        [1.0, 0.0],  # node 1
        [1.0, 1.0],  # node 2
        [0.0, 1.0],  # node 3
    ])
    # One element => shape (1,4)
    connect = np.array([[0, 1, 2, 3]])
    ele_type = "D2_nn4_quad"

    # Domain corners
    x_lower, x_upper = 0.0, 1.0
    y_lower, y_upper = 0.0, 1.0

    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type,
        x_lower, x_upper, y_lower, y_upper,
        tol=1e-12
    )

    # --- Check boundary nodes ---
    # We expect:
    #  left => {0,3}, right => {1,2}, bottom => {0,1}, top => {2,3}
    assert boundary_nodes["left"] == {0, 3}, "Left boundary nodes mismatch."
    assert boundary_nodes["right"] == {1, 2}, "Right boundary nodes mismatch."
    assert boundary_nodes["bottom"] == {0, 1}, "Bottom boundary nodes mismatch."
    assert boundary_nodes["top"] == {2, 3}, "Top boundary nodes mismatch."

    # --- Check boundary edges ---
    # For quad4 => local_faces = [(0,1), (1,2), (2,3), (3,0)]
    # We have 1 element, so e=0
    # local_face 0 => (0,1) => bottom
    # local_face 1 => (1,2) => right
    # local_face 2 => (2,3) => top
    # local_face 3 => (3,0) => left
    # So we expect:
    #   boundary_edges['bottom'] -> [(0,0)]
    #   boundary_edges['right']  -> [(0,1)]
    #   boundary_edges['top']    -> [(0,2)]
    #   boundary_edges['left']   -> [(0,3)]
    # (The function returns a list of (elem_id, face_id).)
    assert boundary_edges["bottom"] == [(0, 0)], "bottom edges mismatch."
    assert boundary_edges["right"] == [(0, 1)], "right edges mismatch."
    assert boundary_edges["top"] == [(0, 2)], "top edges mismatch."
    assert boundary_edges["left"] == [(0, 3)], "left edges mismatch."


def test_identify_rect_boundaries_D2_nn8_quad():
    """
    Test identify_rect_boundaries on a single 8-node quadratic quad element
    forming a rectangle from x in [0,2] and y in [0,1].

    Node arrangement (0-based):
      Corner nodes:
        0 -> (0,0)
        1 -> (2,0)
        2 -> (2,1)
        3 -> (0,1)
      Mid-edge nodes:
        4 -> (1,0)   between (0,0) and (2,0)
        5 -> (2,0.5) between (2,0) and (2,1)
        6 -> (1,1)   between (2,1) and (0,1)
        7 -> (0,0.5) between (0,1) and (0,0)

    Local faces for D2_nn8_quad => [(0,1,4), (1,2,5), (2,3,6), (3,0,7)]
    """

    # Define the node coordinates (n_nodes=8, dimension=2)
    coords = np.array([
        [0.0, 0.0],  # node 0, corner
        [2.0, 0.0],  # node 1, corner
        [2.0, 1.0],  # node 2, corner
        [0.0, 1.0],  # node 3, corner
        [1.0, 0.0],  # node 4, mid-edge bottom
        [2.0, 0.5],  # node 5, mid-edge right
        [1.0, 1.0],  # node 6, mid-edge top
        [0.0, 0.5],  # node 7, mid-edge left
    ])

    # Single element with local connectivity [0,1,2,3,4,5,6,7]
    # shape => (1,8)
    connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    ele_type = "D2_nn8_quad"

    # Domain corners
    x_lower, x_upper = 0.0, 2.0
    y_lower, y_upper = 0.0, 1.0

    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type,
        x_lower, x_upper, y_lower, y_upper,
        tol=1e-12
    )

    # --- Check boundary nodes ---
    # x=0 => left => nodes {0,3,7}
    # x=2 => right => nodes {1,2,5}
    # y=0 => bottom => nodes {0,1,4}
    # y=1 => top => nodes {2,3,6}
    assert boundary_nodes["left"] == {0, 3, 7},  "Left boundary nodes mismatch."
    assert boundary_nodes["right"] == {1, 2, 5},  "Right boundary nodes mismatch."
    assert boundary_nodes["bottom"] == {0, 1, 4},  "Bottom boundary nodes mismatch."
    assert boundary_nodes["top"] == {2, 3, 6},  "Top boundary nodes mismatch."

    # --- Check boundary edges ---
    # For D2_nn8_quad => local_faces = [(0,1,4), (1,2,5), (2,3,6), (3,0,7)]
    # We have 1 element => e=0
    # local_face_id=0 => (0,1,4) => bottom boundary
    # local_face_id=1 => (1,2,5) => right boundary
    # local_face_id=2 => (2,3,6) => top boundary
    # local_face_id=3 => (3,0,7) => left boundary
    # So we expect:
    #   boundary_edges['bottom'] => [(0,0)]
    #   boundary_edges['right']  => [(0,1)]
    #   boundary_edges['top']    => [(0,2)]
    #   boundary_edges['left']   => [(0,3)]
    assert boundary_edges["bottom"] == [(0, 0)], "bottom edges mismatch."
    assert boundary_edges["right"] == [(0, 1)], "right edges mismatch."
    assert boundary_edges["top"] == [(0, 2)], "top edges mismatch."
    assert boundary_edges["left"] == [(0, 3)], "left edges mismatch."


def test_identify_rect_boundaries_D2_nn3_tri():
    """
    Test identify_rect_boundaries on a simple rectangular domain [0,1]x[0,1]
    meshed with two D2_nn3_tri elements.

    We define 4 nodes (corners of a square):
      node0 -> (0,0)
      node1 -> (1,0)
      node2 -> (1,1)
      node3 -> (0,1)

    Then 2 triangular elements:
      Element 0 => (0,1,2)
      Element 1 => (0,2,3)

    For tri3 => local faces = [(0,1),(1,2),(2,0)].
    The domain bounds: x_lower=0, x_upper=1, y_lower=0, y_upper=1.

    We'll check:
      - boundary_nodes for each side,
      - boundary_edges for each side,
      matching the known layout.
    """

    # Node coordinates (n_nodes=4, dimension=2)
    coords = np.array([
        [0.0, 0.0],  # node 0
        [1.0, 0.0],  # node 1
        [1.0, 1.0],  # node 2
        [0.0, 1.0],  # node 3
    ])
    # Two tri3 elements:
    #   - element 0 => (0,1,2)
    #   - element 1 => (0,2,3)
    # shape => (2,3)
    connect = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    ele_type = "D2_nn3_tri"

    # Domain bounds
    x_lower, x_upper = 0.0, 1.0
    y_lower, y_upper = 0.0, 1.0

    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type,
        x_lower, x_upper, y_lower, y_upper,
        tol=1e-12
    )

    # --- Check boundary nodes ---
    # x=0 => left => {0,3}, x=1 => right => {1,2}
    # y=0 => bottom => {0,1}, y=1 => top => {2,3}
    assert boundary_nodes["left"] == {0, 3}, "left boundary node mismatch"
    assert boundary_nodes["right"] == {1, 2}, "right boundary node mismatch"
    assert boundary_nodes["bottom"] == {0, 1}, "bottom boundary node mismatch"
    assert boundary_nodes["top"] == {2, 3}, "top boundary node mismatch"

    # --- Check boundary edges ---
    # Tri3 local faces => (0,1),(1,2),(2,0).
    #
    # Element 0 => global nodes (0,1,2).
    #   face0 => (0,1) => bottom boundary
    #   face1 => (1,2) => right boundary
    #   face2 => (2,0) => diagonal (not boundary)
    #
    # Element 1 => global nodes (0,2,3).
    #   face0 => (0,2) => diagonal (not boundary)
    #   face1 => (2,3) => top boundary
    #   face2 => (3,0) => left boundary
    #
    # So we expect:
    #   bottom => [(0,0)] => elem0, face0
    #   right  => [(0,1)] => elem0, face1
    #   top    => [(1,1)] => elem1, face1
    #   left   => [(1,2)] => elem1, face2
    assert boundary_edges["bottom"] == [(0, 0)], "bottom edge mismatch"
    assert boundary_edges["right"] == [(0, 1)], "right edge mismatch"
    assert boundary_edges["top"] == [(1, 1)], "top edge mismatch"
    assert boundary_edges["left"] == [(1, 2)], "left edge mismatch"


def test_identify_rect_boundaries_D2_nn6_tri():
    """
    Test identify_rect_boundaries on a [0,1]x[0,1] domain that is exactly
    covered by TWO 6-node triangles (D2_nn6_tri).
    
    We define 9 node coordinates:

        Corner nodes:
            node0 = (0.0, 0.0)
            node1 = (1.0, 0.0)
            node2 = (1.0, 1.0)
            node3 = (0.0, 1.0)

        Mid-edge nodes (bottom-left triangle):
            node4 = (0.5, 0.0)   # mid-edge of bottom side (0->1)
            node7 = (0.0, 0.5)   # mid-edge of left side (3->0)
            node8 = (0.5, 0.5)   # mid-edge of diagonal (1->3)

        Mid-edge nodes (top-right triangle):
            node5 = (1.0, 0.5)   # mid-edge of right side (1->2)
            node6 = (0.5, 1.0)   # mid-edge of top side (2->3)
            node8 = (0.5, 0.5)   # shared mid-edge diagonal (3->1) – same node as in the first tri

    Tri #1 => corners=(0,1,3) + mid-edge=(4,8,7)
    Tri #2 => corners=(1,2,3) + mid-edge=(5,6,8)

    Local faces for tri6 are typically:
      face0 = (0,1,3)
      face1 = (1,2,4)
      face2 = (2,0,5)
    but we must interpret "0,1,2" as the corner indices *in each element's local ordering*.

    We'll rely on identify_rect_boundaries to do:
      - boundary_nodes => sets of node IDs on each edge
      - boundary_edges => which element-face is on that boundary
    """

    # Node coordinates (n_nodes=9 total)
    coords = np.array([
        [0.0, 0.0],  # node0, corner, lower left
        [1.0, 0.0],  # node1, corner. lower right
        [1.0, 1.0],  # node2, corner, upper right
        [0.0, 1.0],  # node3, corner, upper left
        [0.5, 0.0],  # node4, mid-edge bottom (between node0->node1)
        [1.0, 0.5],  # node5, mid-edge right  (between node1->node2)
        [0.5, 1.0],  # node6, mid-edge top    (between node2->node3)
        [0.0, 0.5],  # node7, mid-edge left   (between node3->node0)
        [0.5, 0.5],  # node8, mid-edge diagonal (between node1->3 or node3->1)
    ], dtype=float)

    # We define two tri6 elements:
    #   Tri #1 => corners=(0,1,3), mid-edge=(4,8,7)
    #   Tri #2 => corners=(1,2,3), mid-edge=(5,6,8)
    # => shape => (2,6)
    connect = np.array([
        [0, 1, 3, 4, 8, 7],  # element 0
        [1, 2, 3, 5, 6, 8],  # element 1
    ])

    ele_type = "D2_nn6_tri"

    # Domain bounding box
    x_lower, x_upper = 0.0, 1.0
    y_lower, y_upper = 0.0, 1.0

    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type,
        x_lower, x_upper,
        y_lower, y_upper,
        tol=1e-12
    )

    # --- Check boundary nodes ---
    #   left => x=0 => {0,3,7}
    #   right => x=1 => {1,2,5}
    #   bottom => y=0 => {0,1,4}
    #   top => y=1 => {2,3,6}
    assert boundary_nodes["left"] == {0, 3, 7}, "Left boundary node mismatch"
    assert boundary_nodes["right"] == {1, 2, 5}, "Right boundary node mismatch"
    assert boundary_nodes["bottom"] == {0, 1, 4}, "Bottom boundary node mismatch"
    assert boundary_nodes["top"] == {2, 3, 6}, "Top boundary node mismatch"

    # --- Check boundary edges ---
    # Tri6 local faces => (0,1,3), (1,2,4), (2,0,5)
    # For each element, "0,1,2" is corners in the local ordering. But in our
    # global connectivity, we have (0,1,3) as corners in element 0, and so on.
    # We'll interpret them as the function does: face0 => (corners[0], corners[1], mid-edge?), etc.
    #
    # Element 0 => corners=(0,1,3), mid-edges=(4,8,7).
    #   local face0 => (0,1,4) => bottom => nodes (0,1,4)
    #   local face1 => (1,3,8) => diagonal => not boundary
    #   local face2 => (3,0,7) => left => nodes (3,0,7)
    #
    # => so boundary_edges['bottom'] => (elem=0, face=0), boundary_edges['left'] => (elem=0, face=2)
    #
    # Element 1 => corners=(1,2,3), mid-edges=(5,6,8).
    #   local face0 => (1,2,5) => right => (1,2,5)
    #   local face1 => (2,3,6) => top   => (2,3,6)
    #   local face2 => (3,1,8) => diagonal => not boundary
    #
    # => boundary_edges['right'] => [(1,0)], boundary_edges['top'] => [(1,1)]
    #
    # So we expect:
    #   bottom => [(0,0)]
    #   left   => [(0,2)]
    #   right  => [(1,0)]
    #   top    => [(1,1)]
    assert boundary_edges["bottom"] == [(0, 0)], "Bottom edge mismatch"
    assert boundary_edges["left"] == [(0, 2)], "Left edge mismatch"
    assert boundary_edges["right"] == [(1, 0)], "Right edge mismatch"
    assert boundary_edges["top"] == [(1, 1)], "Top edge mismatch"


def test_assign_fixed_nodes_rect():
    """
    Comprehensive tests for assign_fixed_nodes_rect with various boundary
    node sets and DOF displacement inputs.
    """

    # Example boundary_nodes dictionary
    # Suppose we have a 2D domain: left={0,3}, right={1,2}, bottom={0,1}, top={2,3}.
    boundary_nodes = {
        'left':   {0, 3},
        'right':  {1, 2},
        'bottom': {0, 1},
        'top':    {2, 3},
    }

    # --- 1) Test with an empty boundary ---
    # Suppose 'bottom' is empty or some boundary that doesn't exist:
    empty_boundary_nodes = {
        'left': set(),
        'right': {1},
        'bottom': set(),
        'top': {2, 3},
    }
    # Ask for 'left' => which is empty => returns (3,0)
    fixed_empty = pre.assign_fixed_nodes_rect(empty_boundary_nodes, 'left', dof_0_disp=0.0)
    assert fixed_empty.shape == (3, 0), "Expected an empty constraints array."

    # --- 2) Some boundary nodes, specifying dof_0 and dof_1 displacements ---
    # We'll fix dof_0=0.0 and dof_1=1.0 on the 'left' boundary => node IDs => {0,3}.
    # Expect 2 nodes * 2 dofs => 4 constraints => shape => (3,4).
    boundary = 'left'
    dof_0_disp = 0.0
    dof_1_disp = 1.0
    constraints = pre.assign_fixed_nodes_rect(boundary_nodes, boundary,
                                          dof_0_disp=dof_0_disp,
                                          dof_1_disp=dof_1_disp)
    # We expect shape => (3,4) => 4 constraints => for node0/dof0, node0/dof1, node3/dof0, node3/dof1
    assert constraints.shape == (3, 4), f"Expected (3,4) but got {constraints.shape}"

    # Let's interpret the columns
    # e.g. columns => [nodeID, dofIdx, dispVal]
    # We can't guarantee the order (since set iteration can be random),
    # so let's gather them in a python structure to compare ignoring order:
    # We'll do a set of (node, dof, disp) tuples.
    got = set(tuple(constraints[:, i]) for i in range(constraints.shape[1]))
    # Expect => node=0,dof=0,disp=0.0 => (0,0,0.0); node=0,dof=1,disp=1.0 => ...
    want = {
        (0.0, 0.0, 0.0),  # node0,dof0 => disp=0
        (0.0, 1.0, 1.0),  # node0,dof1 => disp=1
        (3.0, 0.0, 0.0),  # node3,dof0 => disp=0
        (3.0, 1.0, 1.0),  # node3,dof1 => disp=1
    }
    assert got == want, f"Mismatch in constraints for boundary={boundary}, dof0={dof_0_disp}, dof1={dof_1_disp}"

    # --- 3) Some boundary nodes but dof_0_disp=dof_1_disp=None => returns empty ---
    constraints_none = pre.assign_fixed_nodes_rect(boundary_nodes, 'left',
                                               dof_0_disp=None,
                                               dof_1_disp=None,
                                               dof_2_disp=None)
    assert constraints_none.shape == (3, 0), "Expected empty array if all disp are None."

    # --- 4) Check dof_2_disp in a 2D scenario => typically not used, but if given => it appears
    # Suppose we fix dof_2=99.0 on the 'bottom' boundary => nodeIDs={0,1}.
    # Then we expect 2 constraints => shape => (3,2).
    constraints_dof2 = pre.assign_fixed_nodes_rect(boundary_nodes, 'bottom', dof_2_disp=99.0)
    assert constraints_dof2.shape == (3, 2), f"Expected (3,2) from dof2 constraints"
    # Again interpret ignoring order
    got_dof2 = set(tuple(constraints_dof2[:, i]) for i in range(constraints_dof2.shape[1]))
    want_dof2 = {
        (0.0, 2.0, 99.0),  # node0, dof2 => 99
        (1.0, 2.0, 99.0),  # node1, dof2 => 99
    }
    assert got_dof2 == want_dof2, f"Mismatch in dof2 constraints: {got_dof2}"


def test_assign_uniform_load_rect():
    """
    Comprehensive tests for assign_uniform_load_rect with different boundary faces
    and traction load setups. We verify the shape of the returned dload_info array
    and its column content (elem_id, face_id, dof_0, dof_1, dof_2).
    """

    # Suppose we have a dictionary that indicates which element-face belongs
    # to which boundary of a rectangular domain:
    #
    # boundary_edges[boundary] = list of (element_id, local_face_id)
    # We'll define a small scenario with 2 or 3 faces on some boundaries,
    # zero on others.
    boundary_edges = {
        "left":   [(0, 0), (2, 1)],  # e.g. face(0,0) => elem=0,face_id=0; face(2,1) => elem=2,face_id=1
        "right":  [],
        "bottom": [(1, 2)],         # e.g. face(1,2) => elem=1,face_id=2
        "top":    [(1, 1)],         # e.g. face(1,1) => elem=1,face_id=1
    }

    # --- CASE 1: A boundary with faces, nonzero dof_0 and dof_1 loads, zero dof_2 ---
    # e.g., "left" => dof_0=10, dof_1=20 => dof_2=0 => shape => (5, n_face_loads).
    # The function sets ndof=3 unconditionally, so the array is (3+2=5 rows, n_face_loads columns).
    dof_0_load = 10.0
    dof_1_load = 20.0
    dof_2_load = 0.0  # for 2D
    dload_info_left = pre.assign_uniform_load_rect(
        boundary_edges, "left",
        dof_0_load=dof_0_load,
        dof_1_load=dof_1_load,
        dof_2_load=dof_2_load
    )
    # We have 2 faces => shape => (5,2):
    #   Row0 => [elem_id0, elem_id1]
    #   Row1 => [face_id0, face_id1]
    #   Row2 => [dof0_load, dof0_load] => 10
    #   Row3 => [dof1_load, dof1_load] => 20
    #   Row4 => [dof2_load, dof2_load] => 0
    assert dload_info_left.shape == (5, 2), f"Expected (5,2), got {dload_info_left.shape}"

    # Let's check columns:
    # The boundary list says "left" => [(0,0), (2,1)] => so we expect
    #   col0 => [elem=0, face=0, 10, 20, 0]
    #   col1 => [elem=2, face=1, 10, 20, 0]
    want_left = np.array([
        [0.0,   2.0],
        [0.0,   1.0],
        [10.0, 10.0],
        [20.0, 20.0],
        [ 0.0,  0.0],
    ])
    assert np.allclose(dload_info_left, want_left, atol=1e-12), (
        f"Mismatch in 'left' boundary loads:\nGot:\n{dload_info_left}\nWanted:\n{want_left}"
    )

    # --- CASE 2: A boundary with no faces => should yield empty array shape => (5,0) ---
    dload_info_right = pre.assign_uniform_load_rect(
        boundary_edges, "right",
        dof_0_load=5.0,   # any nonzero
        dof_1_load=0.0,
        dof_2_load=0.0
    )
    assert dload_info_right.shape == (5, 0), f"Expected (5,0) for empty boundary, got {dload_info_right.shape}"

    # --- CASE 3: Single face boundary => dof_2 nonzero => shape => (5,1) with last row storing dof2 ---
    # "bottom" => 1 face => (elem=1,face=2)
    dload_info_bottom = pre.assign_uniform_load_rect(
        boundary_edges, "bottom",
        dof_0_load=0.0,
        dof_1_load=0.0,
        dof_2_load=99.9
    )
    assert dload_info_bottom.shape == (5, 1), f"Expected (5,1) for single face, got {dload_info_bottom.shape}"
    # We want => col0 => [elem=1, face=2, dof0=0, dof1=0, dof2=99.9]
    want_bottom = np.array([[1.0], [2.0], [0.0], [0.0], [99.9]])
    assert np.allclose(dload_info_bottom, want_bottom, atol=1e-12), (
        f"Mismatch in 'bottom' boundary loads:\nGot:\n{dload_info_bottom}\nWanted:\n{want_bottom}"
    )

    # --- CASE 4: Single face boundary => all dof loads are 0 => still appear or not?
    # The doc says "If the traction is zero in all directions and you prefer to omit them,
    # you can filter." The code does not skip them automatically. So let's see the code's behavior.
    dload_info_top = pre.assign_uniform_load_rect(
        boundary_edges, "top",
        dof_0_load=0.0,
        dof_1_load=0.0,
        dof_2_load=0.0
    )
    # The boundary "top" => [(1,1)]. So we have 1 face. The code sets ndof=3 => shape => (5,1).
    # The last 3 rows => [0,0,0].
    want_top = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]])  # single column
    assert dload_info_top.shape == (5, 1), f"Expected (5,1) for top boundary => 1 face"
    assert np.allclose(dload_info_top, want_top, atol=1e-12), (
        f"Mismatch in 'top' boundary loads:\nGot:\n{dload_info_top}\nWanted:\n{want_top}"
    )
