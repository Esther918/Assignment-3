import numpy as np


###########################################################
# ELEMENT TYPE INFORMATION -- WRAPPER FUNCTION
###########################################################

def element_info(ele_type: str):
    """
    Returns the number of coordinates, number of degrees of freedom (DOFs),
    and number of element nodes for a given finite element type.

    Parameters:
        ele_type (str): The element type identifier.

    Returns:
        tuple:
            - int: Number of coordinates (1 for 1D, 2 for 2D, 3 for 3D).
            - int: Number of degrees of freedom (same as number of coordinates).
            - int: Number of element nodes.

    Raises:
        ValueError: If ele_type is not recognized.
    """
    element_data = {
        "D1_nn2": (1, 1, 2),  # 1D element with 2 nodes
        "D1_nn3": (1, 1, 3),  # 1D element with 3 nodes
        "D2_nn3_tri": (2, 2, 3),  # 2D triangular element with 3 nodes
        "D2_nn6_tri": (2, 2, 6),  # 2D triangular element with 6 nodes
        "D2_nn4_quad": (2, 2, 4),  # 2D quadrilateral element with 4 nodes
        "D2_nn8_quad": (2, 2, 8)  # 2D quadrilateral element with 8 nodes
    }

    if ele_type not in element_data:
        raise ValueError(f"Unknown element type: {ele_type}")

    return element_data[ele_type]


###########################################################
# SHAPE FUNCTIONS AND DERIVATIVES -- WRAPPER FUNCTIONS
###########################################################

def shape_fcn(ele_type: str, xi: np.ndarray) -> np.ndarray:
    """
    Evaluate the shape functions for a given finite element type at natural coordinates.

    Parameters
    ----------
    ele_type : str
        The element type identifier. Supported types include:
        - "D1_nn2" : 1D linear element (2 nodes)
        - "D1_nn3" : 1D quadratic element (3 nodes)
        - "D2_nn3_tri" : 2D linear triangle (3 nodes)
        - "D2_nn6_tri" : 2D quadratic triangle (6 nodes)
        - "D2_nn4_quad" : 2D bilinear quadrilateral (4 nodes)
        - "D2_nn8_quad" : 2D quadratic quadrilateral (8 nodes)

    xi : np.ndarray
        A NumPy array representing the natural coordinates where the shape
        functions should be evaluated.

    Returns
    -------
    N : np.ndarray
        A NumPy array containing the evaluated shape functions at xi.

    Raises
    ------
    ValueError
        If the element type is not recognized.

    Notes
    -----
    - This function provides a clean interface for evaluating shape functions
      without needing to call individual shape function implementations manually.
    - It is used in **finite element analysis (FEA)** for interpolation and
      numerical integration.
    """
    
    # Dictionary mapping element types to shape function implementations
    shape_function_map = {
        "D1_nn2": D1_nn2,
        "D1_nn3": D1_nn3,
        "D2_nn3_tri": D2_nn3_tri,
        "D2_nn6_tri": D2_nn6_tri,
        "D2_nn4_quad": D2_nn4_quad,
        "D2_nn8_quad": D2_nn8_quad
    }

    # Ensure the element type is valid
    if ele_type not in shape_function_map:
        raise ValueError(f"Unsupported element type '{ele_type}'. "
                         "Supported types: " + ", ".join(shape_function_map.keys()))

    # Evaluate the shape function for the given element type
    return shape_function_map[ele_type](xi)


def shape_fcn_derivative(ele_type: str, xi: np.ndarray) -> np.ndarray:
    """
    Evaluate the shape function derivatives for a given finite element type at natural coordinates.

    Parameters
    ----------
    ele_type : str
        The element type identifier. Supported types include:
        - "D1_nn2" : 1D linear element (2 nodes)
        - "D1_nn3" : 1D quadratic element (3 nodes)
        - "D2_nn3_tri" : 2D linear triangle (3 nodes)
        - "D2_nn6_tri" : 2D quadratic triangle (6 nodes)
        - "D2_nn4_quad" : 2D bilinear quadrilateral (4 nodes)
        - "D2_nn8_quad" : 2D quadratic quadrilateral (8 nodes)

    xi : np.ndarray
        A NumPy array representing the natural coordinates where the shape
        function derivatives should be evaluated.

    Returns
    -------
    dN_dxi : np.ndarray
        A NumPy array containing the evaluated shape function derivatives at xi.
        - Each **row** corresponds to a **node**.
        - Each **column** corresponds to derivatives with respect to **ξ (0), η (1), and ζ (2)**.

    Raises
    ------
    ValueError
        If the element type is not recognized.

    Notes
    -----
    - This function provides a clean interface for evaluating shape function derivatives
      without needing to call individual implementations manually.
    - It is used in **finite element analysis (FEA)** for constructing the **B-matrix**,
      which relates strain to nodal displacements.
    """
    
    # Dictionary mapping element types to shape function derivative implementations
    shape_function_derivative_map = {
        "D1_nn2": D1_nn2_dxi,
        "D1_nn3": D1_nn3_dxi,
        "D2_nn3_tri": D2_nn3_tri_dxi,
        "D2_nn6_tri": D2_nn6_tri_dxi,
        "D2_nn4_quad": D2_nn4_quad_dxi,
        "D2_nn8_quad": D2_nn8_quad_dxi,
    }

    # Ensure the element type is valid
    if ele_type not in shape_function_derivative_map:
        raise ValueError(f"Unsupported element type '{ele_type}'. "
                         "Supported types: " + ", ".join(shape_function_derivative_map.keys()))

    # Evaluate the shape function derivative for the given element type
    return shape_function_derivative_map[ele_type](xi)

###########################################################
# GAUSSIAN INTEGRATION POINTS AND WEIGHTS -- WRAPPER FUNCTION
###########################################################


def integration_info(ele_type: str):
    """
    Returns the number of integration points, integration points, and integration weights
    for a given finite element type.

    Parameters:
        ele_type (str): The element type identifier.

    Returns:
        tuple:
            - int: Number of integration points.
            - np.ndarray: Integration points (shape depends on element type).
            - np.ndarray: Integration weights (num_pts, 1).

    Raises:
        ValueError: If ele_type is not recognized.
    """
    element_data = {
        "D1_nn2": (2, gauss_points_1d(2), gauss_weights_1d(2)),
        "D1_nn3": (3, gauss_points_1d(3), gauss_weights_1d(3)),
        "D2_nn3_tri": (1, gauss_points_2d_triangle(1), gauss_weights_2d_triangle(1)),
        "D2_nn6_tri": (3, gauss_points_2d_triangle(3), gauss_weights_2d_triangle(3)),
        "D2_nn4_quad": (4, gauss_points_2d_quad(4), gauss_weights_2d_quad(4)),
        "D2_nn8_quad": (9, gauss_points_2d_quad(9), gauss_weights_2d_quad(9))
    }

    if ele_type not in element_data:
        raise ValueError(f"Unknown element type: {ele_type}")

    return element_data[ele_type]


###########################################################
###########################################################
# FACE NODES
###########################################################
###########################################################


def face_info(ele_type: str, face: int = 0):
    """
    Return information about the specified face of a finite element, including:
      - The 0-based node indices on that face.
      - The number of face nodes (for that element type).
      - The element type of the face (i.e., a lower-dimensional element type).

    Parameters
    ----------
    ele_type : str
        The high-dimensional element type identifier, e.g. 'D2_nn3_tri', 'D3_nn8_hex'.
    face : int
        0-based face (or edge) index. The valid range depends on the element type.

    Returns
    -------
    nodes_on_face : list of int
        0-based node indices that define the face.
        with the default value of 0, it is assumed that this information is not needed.
    num_face_nodes : int
        The number of nodes on each face of this element type.
    face_element_type : str or None
        The element type corresponding to the lower-dimensional face.
        For instance, a 3D tetra face might be a 2D triangle element type.
        Some 1D elements have no lower dimension; in that case, this returns None.

    Raises
    ------
    ValueError
        If the element type is not recognized or if 'face' is out of range.
    """

    # Dictionary that maps each element type to:
    #   (face_node_func, num_face_nodes, face_element_type).
    # 'face_element_type' is the "surface" element type that a face forms.
    face_info_map = {
        "D1_nn2":      (get_face_nodes_D1_nn2, 1, None),
        "D1_nn3":      (get_face_nodes_D1_nn3, 1, None),
        "D2_nn3_tri":  (get_face_nodes_D2_nn3_tri, 2, "D1_nn2"),
        "D2_nn6_tri":  (get_face_nodes_D2_nn6_tri, 3, "D1_nn3"),
        "D2_nn4_quad": (get_face_nodes_D2_nn4_quad, 2, "D1_nn2"),
        "D2_nn8_quad": (get_face_nodes_D2_nn8_quad, 3, "D1_nn3"),
    }

    if ele_type not in face_info_map:
        raise ValueError(f"Unknown element type: {ele_type}")

    face_func, num_face_nodes, face_elem_type = face_info_map[ele_type]

    # Compute the list of node indices on the specified face
    nodes_on_face = face_func(face)

    # Validate that the length of nodes_on_face equals num_face_nodes
    if len(nodes_on_face) != num_face_nodes:
        raise RuntimeError(
            f"Mismatch for {ele_type} on face={face}: expected "
            f"{num_face_nodes} nodes but got {len(nodes_on_face)}."
        )

    return face_elem_type, num_face_nodes, nodes_on_face


###########################################################
###########################################################
# SHAPE FUNCTIONS AND DERIVATIVES -- 1D
###########################################################
###########################################################

def D1_nn2(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 1D shape functions for a two-node element in natural coordinates.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (1,)) representing the natural coordinate xi.
        xi is in the range [-1, 1].

    Returns
    -------
    N : np.ndarray
        A (2,1) NumPy array containing the evaluated shape functions at xi.

    Notes
    -----
    - D1 refers to a **1D** element.
    - nn2 refers to **2 nodal values** (linear element).
    - The shape functions are defined as:
        N1(xi) = 0.5 * (1 + xi)
        N2(xi) = 0.5 * (1 - xi)
    """
    N = np.zeros((2, 1))
    N[0, 0] = 0.5 * (1.0 - xi[0])
    N[1, 0] = 0.5 * (1.0 + xi[0])
    return N


def D1_nn2_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the 1D shape functions for a two-node element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (1,)) representing the natural coordinate xi.
        xi is in the range [-1, 1] (not used, as derivatives are constant).

    Returns
    -------
    dN_dxi : np.ndarray
        A (2,1) NumPy array containing the derivatives of the shape functions
        with respect to the natural coordinate xi.

    Notes
    -----
    - D1 refers to a **1D** element.
    - nn2 refers to **2 nodal values** (linear element).
    - The shape function derivatives are:
        dN1/dxi = -0.5
        dN2/dxi = 0.5
    """
    dN_dxi = np.zeros((2, 1))
    dN_dxi[0, 0] = -0.5
    dN_dxi[1, 0] = 0.5
    return dN_dxi


###########################################################


def D1_nn3(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 1D quadratic shape functions for a three-node element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (1,)) representing the natural coordinate xi.
        xi is in the range [-1, 1].

    Returns
    -------
    N : np.ndarray
        A (3,1) NumPy array containing the evaluated shape functions at xi.

    Notes
    -----
    - D1 refers to a **1D** element.
    - nn3 refers to **3 nodal values** (quadratic element).
    - The quadratic shape functions are:
        N1(xi) = -0.5 * xi * (1 - xi)
        N2(xi) =  0.5 * xi * (1 + xi)
        N3(xi) =  (1 - xi) * (1 + xi)
    """
    N = np.zeros((3, 1))
    N[0, 0] = -0.5 * xi[0] * (1.0 - xi[0])
    N[1, 0] = 0.5 * xi[0] * (1.0 + xi[0])
    N[2, 0] = (1.0 - xi[0]) * (1.0 + xi[0])
    return N


def D1_nn3_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the 1D quadratic shape functions for a three-node element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (1,)) representing the natural coordinate xi.
        xi is in the range [-1, 1].

    Returns
    -------
    dN_dxi : np.ndarray
        A (3,1) NumPy array containing the derivatives of the shape functions
        with respect to the natural coordinate xi.

    Notes
    -----
    - D1 refers to a **1D** element.
    - nn3 refers to **3 nodal values** (quadratic element).
    - The shape function derivatives are:
        dN1/dxi = -0.5 + xi
        dN2/dxi =  0.5 + xi
        dN3/dxi = -2.0 * xi
    """
    dN_dxi = np.zeros((3, 1))
    dN_dxi[0, 0] = -0.5 + xi[0]
    dN_dxi[1, 0] =  0.5 + xi[0]
    dN_dxi[2, 0] = -2.0 * xi[0]
    return dN_dxi


###########################################################
###########################################################
# SHAPE FUNCTIONS AND DERIVATIVES -- 2D
###########################################################
###########################################################

def D2_nn3_tri(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 2D linear shape functions for a three-node triangular element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [0,1], defining the local triangular coordinate system.

    Returns
    -------
    N : np.ndarray
        A (3,1) NumPy array containing the evaluated shape functions at (ξ, η).

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn3 refers to **3 nodal values** (linear triangular element).
    - The shape functions for a standard triangular element are:
        N1(ξ, η) = ξ
        N2(ξ, η) = η
        N3(ξ, η) = 1 - ξ - η
    - These shape functions are used in **finite element analysis (FEA)**
      to interpolate values within a triangular element.
    """
    N = np.zeros((3, 1))
    N[0, 0] = xi[0]
    N[1, 0] = xi[1]
    N[2, 0] = 1.0 - xi[0] - xi[1]
    return N


def D2_nn3_tri_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the 2D linear shape functions for a three-node triangular element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [0,1], defining the local triangular coordinate system.
        (Not directly used, as the derivatives of linear shape functions are constant.)

    Returns
    -------
    dN_dxi : np.ndarray
        A (3,2) NumPy array containing the derivatives of the shape functions
        with respect to the natural coordinates (ξ, η).
        Each row corresponds to a node, and columns correspond to derivatives
        with respect to ξ and η.

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn3 refers to **3 nodal values** (linear triangular element).
    - The shape function derivatives are:
        dN1/dξ =  1.0    , dN1/dη =  0.0
        dN2/dξ =  0.0    , dN2/dη =  1.0
        dN3/dξ = -1.0    , dN3/dη = -1.0
    - These derivatives are used in **finite element analysis (FEA)**
      to compute the strain-displacement matrix (B-matrix).
    """
    dN_dxi = np.zeros((3, 2))
    dN_dxi[0, 0] = 1.0  # dN1/dξ
    dN_dxi[1, 1] = 1.0  # dN2/dη
    dN_dxi[2, 0] = -1.0  # dN3/dξ
    dN_dxi[2, 1] = -1.0  # dN3/dη
    return dN_dxi


###########################################################

def D2_nn6_tri(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 2D quadratic shape functions for a six-node triangular element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [0,1], defining the local triangular coordinate system.

    Returns
    -------
    N : np.ndarray
        A (6,1) NumPy array containing the evaluated quadratic shape functions at (ξ, η).

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn6 refers to **6 nodal values** (quadratic triangular element).
    - The quadratic shape functions are:
        N1(ξ, η) = (2ξ - 1) * ξ
        N2(ξ, η) = (2η - 1) * η
        N3(ξ, η) = (2ξ_c - 1) * ξ_c
        N4(ξ, η) = 4ξη
        N5(ξ, η) = 4ηξ_c
        N6(ξ, η) = 4ξ_cξ
      where ξ_c = 1 - ξ - η (complementary coordinate for the third node).
    - These shape functions are used in **finite element analysis (FEA)**
      to interpolate field variables within a **quadratic triangular element**.
    """
    N = np.zeros((6, 1))
    xic = 1.0 - xi[0] - xi[1]  # Complementary coordinate (ξ_c)
    N[0, 0] = (2.0 * xi[0] - 1.0) * xi[0]
    N[1, 0] = (2.0 * xi[1] - 1.0) * xi[1]
    N[2, 0] = (2.0 * xic - 1.0) * xic
    N[3, 0] = 4.0 * xi[0] * xi[1]
    N[4, 0] = 4.0 * xi[1] * xic
    N[5, 0] = 4.0 * xic * xi[0]
    return N


def D2_nn6_tri_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the partial derivatives of the 2D quadratic shape functions for a
    six-node triangular element (Lagrange P2 triangle).

    Parameters
    ----------
    xi : np.ndarray of shape (2,)
        The natural coordinates (ξ, η), where 0 ≤ ξ, η and ξ + η ≤ 1 in the
        reference triangle coordinate system.

    Returns
    -------
    dN_dxi : np.ndarray of shape (6, 2)
        The derivatives of the shape functions with respect to (ξ, η).
        Each row corresponds to a node, with columns representing:
        - dN/dξ (∂N/∂ξ)
        - dN/dη (∂N/∂η)

    Notes
    -----
    - Define ξ_c = 1 - ξ - η (the complementary barycentric coordinate).
    - Standard quadratic shape functions:
        N1 = ξ(2ξ - 1)
        N2 = η(2η - 1)
        N3 = ξ_c(2ξ_c - 1)
        N4 = 4 ξ η
        N5 = 4 η ξ_c
        N6 = 4 ξ_c ξ
    - Their partial derivatives are computed as:
        ∂N1/∂ξ =  4ξ - 1      , ∂N1/∂η =  0
        ∂N2/∂ξ =  0           , ∂N2/∂η =  4η - 1
        ∂N3/∂ξ =  - (4ξ_c - 1), ∂N3/∂η =  - (4ξ_c - 1)
        ∂N4/∂ξ =  4η          , ∂N4/∂η =  4ξ
        ∂N5/∂ξ = -4η          , ∂N5/∂η =  4(ξ_c - η)
        ∂N6/∂ξ =  4(ξ_c - ξ)  , ∂N6/∂η =  -4ξ
    """
    dN_dxi = np.zeros((6, 2))
    xi_val, eta_val = xi[0], xi[1]
    xic = 1.0 - xi_val - eta_val  # Complementary coordinate ξ_c

    # Corner nodes (primary nodes)
    dN_dxi[0] = [4.0 * xi_val - 1.0, 0.0]  # ∂N1/∂ξ, ∂N1/∂η
    dN_dxi[1] = [0.0, 4.0 * eta_val - 1.0]  # ∂N2/∂ξ, ∂N2/∂η
    dN_dxi[2] = [-(4.0 * xic - 1.0), -(4.0 * xic - 1.0)]  # ∂N3/∂ξ, ∂N3/∂η

    # Mid-edge nodes
    dN_dxi[3] = [4.0 * eta_val, 4.0 * xi_val]  # ∂N4/∂ξ, ∂N4/∂η
    dN_dxi[4] = [-4.0 * eta_val, 4.0 * (xic - eta_val)]  # ∂N5/∂ξ, ∂N5/∂η
    dN_dxi[5] = [4.0 * (xic - xi_val), -4.0 * xi_val]  # ∂N6/∂ξ, ∂N6/∂η

    return dN_dxi


###########################################################

def D2_nn4_quad(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 2D bilinear shape functions for a four-node quadrilateral element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [-1,1], defining the local quadrilateral coordinate system.

    Returns
    -------
    N : np.ndarray
        A (4,1) NumPy array containing the evaluated bilinear shape functions at (ξ, η).

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn4 refers to **4 nodal values** (bilinear quadrilateral element).
    - The bilinear shape functions are:
        N1(ξ, η) = 0.25 * (1 - ξ) * (1 - η)
        N2(ξ, η) = 0.25 * (1 + ξ) * (1 - η)
        N3(ξ, η) = 0.25 * (1 + ξ) * (1 + η)
        N4(ξ, η) = 0.25 * (1 - ξ) * (1 + η)
    - These shape functions are used in **finite element analysis (FEA)**
      to interpolate field variables within a bilinear quadrilateral element.
    """
    N = np.zeros((4, 1))
    N[0, 0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])  # N1
    N[1, 0] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])  # N2
    N[2, 0] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])  # N3
    N[3, 0] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])  # N4
    return N


def D2_nn4_quad_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the 2D bilinear shape functions for a four-node quadrilateral element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [-1,1], defining the local quadrilateral coordinate system.

    Returns
    -------
    dN_dxi : np.ndarray
        A (4,2) NumPy array containing the derivatives of the shape functions
        with respect to the natural coordinates (ξ, η).
        Each row corresponds to a node, and columns correspond to derivatives
        with respect to ξ and η.

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn4 refers to **4 nodal values** (bilinear quadrilateral element).
    - The derivatives of the bilinear shape functions are:
        dN1/dξ = -0.25 * (1 - η) , dN1/dη = -0.25 * (1 - ξ)
        dN2/dξ =  0.25 * (1 - η) , dN2/dη = -0.25 * (1 + ξ)
        dN3/dξ =  0.25 * (1 + η) , dN3/dη =  0.25 * (1 + ξ)
        dN4/dξ = -0.25 * (1 + η) , dN4/dη =  0.25 * (1 - ξ)
    - These derivatives are used in **finite element analysis (FEA)**
      to compute the strain-displacement matrix (B-matrix).
    """
    dN_dxi = np.zeros((4, 2))
    dN_dxi[0, 0] = -0.25 * (1.0 - xi[1])  # dN1/dξ
    dN_dxi[0, 1] = -0.25 * (1.0 - xi[0])  # dN1/dη
    dN_dxi[1, 0] = 0.25 * (1.0 - xi[1])  # dN2/dξ
    dN_dxi[1, 1] = -0.25 * (1.0 + xi[0])  # dN2/dη
    dN_dxi[2, 0] = 0.25 * (1.0 + xi[1])  # dN3/dξ
    dN_dxi[2, 1] = 0.25 * (1.0 + xi[0])  # dN3/dη
    dN_dxi[3, 0] = -0.25 * (1.0 + xi[1])  # dN4/dξ
    dN_dxi[3, 1] = 0.25 * (1.0 - xi[0])  # dN4/dη
    return dN_dxi


###########################################################

def D2_nn8_quad(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 2D quadratic shape functions for an eight-node quadrilateral element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [-1,1], defining the local quadrilateral coordinate system.

    Returns
    -------
    N : np.ndarray
        A (8,1) NumPy array containing the evaluated quadratic shape functions at (ξ, η).

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn8 refers to **8 nodal values** (quadratic quadrilateral element).
    - The quadratic shape functions for an eight-node quadrilateral element are:
        N1(ξ, η) = -0.25 * (1 - ξ) * (1 - η) * (1 + ξ + η)
        N2(ξ, η) =  0.25 * (1 + ξ) * (1 - η) * (ξ - η - 1)
        N3(ξ, η) =  0.25 * (1 + ξ) * (1 + η) * (ξ + η - 1)
        N4(ξ, η) =  0.25 * (1 - ξ) * (1 + η) * (η - ξ - 1)
        N5(ξ, η) =  0.5 * (1 - ξ²) * (1 - η)
        N6(ξ, η) =  0.5 * (1 + ξ) * (1 - η²)
        N7(ξ, η) =  0.5 * (1 - ξ²) * (1 + η)
        N8(ξ, η) =  0.5 * (1 - ξ) * (1 - η²)
    - These shape functions are used in **finite element analysis (FEA)**
      to interpolate field variables within a **quadratic quadrilateral element**.
    """
    N = np.zeros((8, 1))
    N[0, 0] = -0.25 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[0] + xi[1])  # N1
    N[1, 0] =  0.25 * (1.0 + xi[0]) * (1.0 - xi[1]) * (xi[0] - xi[1] - 1.0)  # N2
    N[2, 0] =  0.25 * (1.0 + xi[0]) * (1.0 + xi[1]) * (xi[0] + xi[1] - 1.0)  # N3
    N[3, 0] =  0.25 * (1.0 - xi[0]) * (1.0 + xi[1]) * (xi[1] - xi[0] - 1.0)  # N4
    N[4, 0] =  0.5 * (1.0 - xi[0] * xi[0]) * (1.0 - xi[1])  # N5 (midpoint)
    N[5, 0] =  0.5 * (1.0 + xi[0]) * (1.0 - xi[1] * xi[1])  # N6 (midpoint)
    N[6, 0] =  0.5 * (1.0 - xi[0] * xi[0]) * (1.0 + xi[1])  # N7 (midpoint)
    N[7, 0] =  0.5 * (1.0 - xi[0]) * (1.0 - xi[1] * xi[1])  # N8 (midpoint)
    return N


def D2_nn8_quad_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the 2D quadratic shape functions for an eight-node quadrilateral element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (2,)) representing the natural coordinates (ξ, η).
        ξ and η are in the range [-1,1], defining the local quadrilateral coordinate system.

    Returns
    -------
    dN_dxi : np.ndarray
        A (8,2) NumPy array containing the derivatives of the shape functions
        with respect to the natural coordinates (ξ, η).
        Each row corresponds to a node, and columns correspond to derivatives
        with respect to ξ and η.

    Notes
    -----
    - D2 refers to a **2D** element.
    - nn8 refers to **8 nodal values** (quadratic quadrilateral element).
    - The derivatives of the quadratic shape functions for an eight-node quadrilateral element are:
        dN1/dξ =  0.25 * (1 - η) * (2ξ + η)      , dN1/dη =  0.25 * (1 - ξ) * (ξ + 2η)
        dN2/dξ =  0.25 * (1 - η) * (2ξ - η)      , dN2/dη =  0.25 * (1 + ξ) * (2η - ξ)
        dN3/dξ =  0.25 * (1 + η) * (2ξ + η)      , dN3/dη =  0.25 * (1 + ξ) * (2η + ξ)
        dN4/dξ =  0.25 * (1 + η) * (2ξ - η)      , dN4/dη =  0.25 * (1 - ξ) * (2η - ξ)
        dN5/dξ = -ξ * (1 - η)                    , dN5/dη = -0.5 * (1 - ξ²)
        dN6/dξ =  0.5 * (1 - η²)                 , dN6/dη = -(1 + ξ) * η
        dN7/dξ = -ξ * (1 + η)                    , dN7/dη =  0.5 * (1 - ξ²)
        dN8/dξ = -0.5 * (1 - η²)                 , dN8/dη = -(1 - ξ) * η
    - These derivatives are used in **finite element analysis (FEA)**
      to compute the strain-displacement matrix (B-matrix).
    """
    dN_dxi = np.zeros((8, 2))

    # Corner Nodes
    dN_dxi[0, 0] = 0.25 * (1.0 - xi[1]) * (2.0 * xi[0] + xi[1])  # dN1/dξ
    dN_dxi[0, 1] = 0.25 * (1.0 - xi[0]) * (xi[0] + 2.0 * xi[1])  # dN1/dη
    dN_dxi[1, 0] = 0.25 * (1.0 - xi[1]) * (2.0 * xi[0] - xi[1])  # dN2/dξ
    dN_dxi[1, 1] = 0.25 * (1.0 + xi[0]) * (2.0 * xi[1] - xi[0])  # dN2/dη
    dN_dxi[2, 0] = 0.25 * (1.0 + xi[1]) * (2.0 * xi[0] + xi[1])  # dN3/dξ
    dN_dxi[2, 1] = 0.25 * (1.0 + xi[0]) * (2.0 * xi[1] + xi[0])  # dN3/dη
    dN_dxi[3, 0] = 0.25 * (1.0 + xi[1]) * (2.0 * xi[0] - xi[1])  # dN4/dξ
    dN_dxi[3, 1] = 0.25 * (1.0 - xi[0]) * (2.0 * xi[1] - xi[0])  # dN4/dη

    # Mid-Side Nodes
    dN_dxi[4, 0] = -xi[0] * (1.0 - xi[1])  # dN5/dξ
    dN_dxi[4, 1] = -0.5 * (1.0 - xi[0] * xi[0])  # dN5/dη
    dN_dxi[5, 0] = 0.5 * (1.0 - xi[1] * xi[1])  # dN6/dξ
    dN_dxi[5, 1] = -(1.0 + xi[0]) * xi[1]  # dN6/dη
    dN_dxi[6, 0] = -xi[0] * (1.0 + xi[1])  # dN7/dξ
    dN_dxi[6, 1] = 0.5 * (1.0 - xi[0] * xi[0])  # dN7/dη
    dN_dxi[7, 0] = -0.5 * (1.0 - xi[1] * xi[1])  # dN8/dξ
    dN_dxi[7, 1] = -(1.0 - xi[0]) * xi[1]  # dN8/dη

    return dN_dxi


###########################################################
# GAUSSIAN INTEGRATION POINTS
###########################################################


def gauss_points_1d(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration points for 1D elements.

    Gauss integration points are used for numerical integration in finite element
    analysis. This function provides the standard 1D Gauss points for different
    levels of accuracy:

    - 1-point rule: xi_array = [0] (suitable for linear functions)
    - 2-point rule: xi_array = [-1/sqrt(3), 1/sqrt(3)] (suitable for quadratic functions)
    - 3-point rule: xi_array = [-sqrt(3/5), 0, sqrt(3/5)] (suitable for cubic functions)

    Parameters:
        num_pts (int): Number of Gauss integration points (1, 2, or 3).

    Returns:
        np.ndarray: A (1, num_pts) array containing the Gauss integration points.

    Raises:
        ValueError: If num_pts is not 1, 2, or 3.
    """
    if num_pts not in {1, 2, 3}:
        raise ValueError("num_pts must be 1, 2, or 3.")

    xi_array = np.zeros((1, num_pts))

    if num_pts == 1:
        pass  # xi_array remains as [0] (default in np.zeros)
    elif num_pts == 2:
        sqrt_3 = np.sqrt(3)
        xi_array[0, 0] = -1.0 / sqrt_3
        xi_array[0, 1] = 1.0 / sqrt_3
    elif num_pts == 3:
        sqrt_3_5 = np.sqrt(3.0 / 5.0)
        xi_array[0, 0] = -sqrt_3_5
        xi_array[0, 1] = 0.0
        xi_array[0, 2] = sqrt_3_5

    return xi_array


def gauss_points_2d_triangle(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration points for triangular elements in 2D.

    Gauss integration points are used for numerical integration in finite element
    analysis. This function provides standard 2D Gauss points for a triangular
    reference element.

    Available quadrature rules:
    - 1-point rule (centroid of the triangle).
    - 3-point rule (suitable for linear functions).
    - 4-point rule (adds the centroid for higher accuracy).

    Parameters:
        num_pts (int): Number of Gauss integration points (1, 3, or 4).

    Returns:
        np.ndarray: A (2, num_pts) array containing the Gauss integration points.

    Raises:
        ValueError: If num_pts is not 1, 3, or 4.
    """
    if num_pts not in {1, 3, 4}:
        raise ValueError("num_pts must be 1, 3, or 4.")

    xi_array = np.zeros((2, num_pts))

    if num_pts == 1:
        # Centroid of the reference triangle
        xi_array[:, 0] = [1.0 / 3.0, 1.0 / 3.0]
    elif num_pts == 3:
        # Three-point quadrature rule
        xi_array[:, 0] = [0.6, 0.2]
        xi_array[:, 1] = [0.2, 0.6]
        xi_array[:, 2] = [0.2, 0.2]
    elif num_pts == 4:
        # Four-point quadrature rule (includes centroid)
        xi_array[:, 0] = [1.0 / 3.0, 1.0 / 3.0]
        xi_array[:, 1] = [0.6, 0.2]
        xi_array[:, 2] = [0.2, 0.6]
        xi_array[:, 3] = [0.2, 0.2]

    return xi_array


def gauss_points_2d_quad(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration points for quadrilateral elements in 2D.

    Gauss integration points are used for numerical integration in finite element
    analysis. This function provides standard 2D Gauss points for a quadrilateral
    reference element.

    Available quadrature rules:
    - 1-point rule (center of the element).
    - 4-point rule (suitable for bilinear functions).
    - 9-point rule (higher accuracy for quadratic functions).

    Parameters:
        num_pts (int): Number of Gauss integration points (1, 4, or 9).

    Returns:
        np.ndarray: A (2, num_pts) array containing the Gauss integration points.

    Raises:
        ValueError: If num_pts is not 1, 4, or 9.
    """
    if num_pts not in {1, 4, 9}:
        raise ValueError("num_pts must be 1, 4, or 9.")

    xi_array = np.zeros((2, num_pts))

    if num_pts == 1:
        # Single-point quadrature (center of the reference square)
        xi_array[:, 0] = [0.0, 0.0]
    elif num_pts == 4:
        # Four-point quadrature rule
        sqrt_3_inv = 1.0 / np.sqrt(3)
        xi_array[:, 0] = [-sqrt_3_inv, -sqrt_3_inv]
        xi_array[:, 1] = [sqrt_3_inv, -sqrt_3_inv]
        xi_array[:, 2] = [-sqrt_3_inv, sqrt_3_inv]
        xi_array[:, 3] = [sqrt_3_inv, sqrt_3_inv]
    elif num_pts == 9:
        # Nine-point quadrature rule
        sqrt_3_5 = np.sqrt(3.0 / 5.0)
        xi_array[:, 0] = [sqrt_3_5, sqrt_3_5]
        xi_array[:, 1] = [0.0, sqrt_3_5]
        xi_array[:, 2] = [-sqrt_3_5, sqrt_3_5]
        xi_array[:, 3] = [sqrt_3_5, 0.0]
        xi_array[:, 4] = [0.0, 0.0]
        xi_array[:, 5] = [-sqrt_3_5, 0.0]
        xi_array[:, 6] = [sqrt_3_5, -sqrt_3_5]
        xi_array[:, 7] = [0.0, -sqrt_3_5]
        xi_array[:, 8] = [-sqrt_3_5, -sqrt_3_5]

    return xi_array


###########################################################
# GAUSSIAN INTEGRATION WEIGHTS
###########################################################


def gauss_weights_1d(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration weights for 1D elements.

    Gauss integration weights are used for numerical integration in finite element
    analysis. This function provides standard 1D Gauss weights for different
    levels of accuracy.

    Available quadrature rules:
    - 1-point rule: w = [2.0] (suitable for linear functions).
    - 2-point rule: w = [1.0, 1.0] (suitable for quadratic functions).
    - 3-point rule: w = [5/9, 8/9, 5/9] (suitable for cubic functions).

    Parameters:
        num_pts (int): Number of Gauss integration points (1, 2, or 3).

    Returns:
        np.ndarray: A (num_pts, 1) array containing the Gauss integration weights.

    Raises:
        ValueError: If num_pts is not 1, 2, or 3.
    """
    if num_pts not in {1, 2, 3}:
        raise ValueError("num_pts must be 1, 2, or 3.")

    w_array = np.zeros((num_pts, 1))

    if num_pts == 1:
        w_array[0, 0] = 2.0
    elif num_pts == 2:
        w_array[:, 0] = [1.0, 1.0]
    elif num_pts == 3:
        w_array[:, 0] = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

    return w_array


def gauss_weights_2d_triangle(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration weights for triangular elements in 2D.

    Gauss integration weights are used for numerical integration in finite element
    analysis. This function provides standard 2D Gauss weights for a triangular
    reference element.

    Available quadrature rules:
    - 1-point rule: w = [0.5] (centroid-based integration).
    - 3-point rule: w = [1/6, 1/6, 1/6] (suitable for linear functions).
    - 4-point rule: w = [-27/96, 25/96, 25/96, 25/96] (higher accuracy).

    Parameters:
        num_pts (int): Number of Gauss integration points (1, 3, or 4).

    Returns:
        np.ndarray: A (num_pts, 1) array containing the Gauss integration weights.

    Raises:
        ValueError: If num_pts is not 1, 3, or 4.
    """
    if num_pts not in {1, 3, 4}:
        raise ValueError("num_pts must be 1, 3, or 4.")

    w_array = np.zeros((num_pts, 1))

    if num_pts == 1:
        w_array[0, 0] = 0.5
    elif num_pts == 3:
        w_array[:, 0] = [1.0 / 6.0] * 3
    elif num_pts == 4:
        w_array[:, 0] = [-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0]

    return w_array


def gauss_weights_2d_quad(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration weights for quadrilateral elements in 2D.

    Gauss integration weights are used for numerical integration in finite element
    analysis. This function provides standard 2D Gauss weights for a quadrilateral
    reference element.

    Available quadrature rules:
    - 1-point rule: w = [4.0] (suitable for linear functions).
    - 4-point rule: w = [1.0, 1.0, 1.0, 1.0] (suitable for bilinear functions).
    - 9-point rule: Computed explicitly as the product of 1D weights [5/9, 8/9, 5/9].

    Parameters:
        num_pts (int): Number of Gauss integration points (1, 4, or 9).

    Returns:
        np.ndarray: A (num_pts, 1) array containing the Gauss integration weights.

    Raises:
        ValueError: If num_pts is not 1, 4, or 9.
    """
    if num_pts not in {1, 4, 9}:
        raise ValueError("num_pts must be 1, 4, or 9.")

    if num_pts == 1:
        w_array = np.array([[4.0]])
    elif num_pts == 4:
        w_array = np.ones((4, 1))
    elif num_pts == 9:
        wgp = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        w_array = np.array([
            wgp[0] * wgp[0], wgp[0] * wgp[1], wgp[0] * wgp[2],
            wgp[1] * wgp[0], wgp[1] * wgp[1], wgp[1] * wgp[2],
            wgp[2] * wgp[0], wgp[2] * wgp[1], wgp[2] * wgp[2],
        ]).reshape(9, 1)  # Shape (9, 1) for consistency

    return w_array


###########################################################
# FACE NODES
###########################################################
def get_face_nodes_D1_nn2(face: int):
    """
    Returns the 0-based node indices on the 'face' of a 1D element
    with 2 total nodes (linear).

    In 1D, there are exactly 2 boundary "faces" (really end-nodes):
      face=0 => [0]
      face=1 => [1]

    Parameters
    ----------
    face : int
        0-based index for the boundary (valid: 0..1).

    Returns
    -------
    list of int
        The 0-based node indices on that boundary of the line element.
    """
    faces_map = {
        0: [0],
        1: [1],
    }
    if face not in faces_map:
        raise ValueError("D1_nn2 has faces=0..1 only.")
    return faces_map[face]


def get_face_nodes_D1_nn3(face: int):
    """
    Returns the 0-based node indices on the 'face' of a 1D element 
    with 3 total nodes (2 end nodes + 1 mid-node).
    
    We can define up to 2 'faces' for the boundary ends, though the presence 
    of a mid-node doesn't typically introduce new boundaries. For consistency, 
    you could define:
      - face=0 => left end node [0]
      - face=1 => right end node [1]

    Alternatively, if you consider each node a separate "face," then 
    you'd have 3 faces: face=0 -> [0], face=1 -> [1], face=2 -> [2]. 
    But that’s not typical in finite element boundary contexts.

    Below is a minimal approach that only allows the two physical boundaries. 
    Adjust if your context differs.

    Parameters
    ----------
    face : int
        0-based index for the boundary (0 or 1).

    Returns
    -------
    list of int
        The 0-based node index on that boundary of the line element.
    """
    faces_map = {
        0: [0],
        1: [1],
    }
    if face not in faces_map:
        raise ValueError("D1_nn3 has faces=0..1 only (end nodes).")
    return faces_map[face]


def get_face_nodes_D2_nn3_tri(face: int):
    """
    Returns the 0-based node indices for a 2D 3-node triangle (linear).
    The triangle has 3 edges, so valid face=0..2.

    Edge connectivity (0-based):
      face=0 => [0, 1]
      face=1 => [1, 2]
      face=2 => [2, 0]

    Parameters
    ----------
    face : int
        0-based edge index (0..2).

    Returns
    -------
    list of int
        The node indices on that edge.
    """
    faces_map = {
        0: [0, 1],
        1: [1, 2],
        2: [2, 0],
    }
    if face not in faces_map:
        raise ValueError("D2_nn3_tri has faces=0..2 only.")
    return faces_map[face]


def get_face_nodes_D2_nn6_tri(face: int):
    """
    Returns the 0-based node indices for a 2D 6-node triangle (quadratic).
    The triangle has 3 edges, face=0..2. Each edge has 2 corner nodes + 1 mid-edge node.

    The typical node numbering is:
      corners: 0,1,2
      mid-edges: 3,4,5
    Where edge i has mid-edge node i+3.

    So:
      face=0 => [0, 1, 3]
      face=1 => [1, 2, 4]
      face=2 => [2, 0, 5]

    Parameters
    ----------
    face : int
        0-based edge index (0..2).

    Returns
    -------
    list of int
        The node indices on that edge.
    """
    faces_map = {
        0: [0, 1, 3],
        1: [1, 2, 4],
        2: [2, 0, 5],
    }
    if face not in faces_map:
        raise ValueError("D2_nn6_tri has faces=0..2 only.")
    return faces_map[face]


def get_face_nodes_D2_nn4_quad(face: int):
    """
    Returns the 0-based node indices for a 2D 4-node quadrilateral (linear).
    The quad has 4 edges, face=0..3. Each edge has 2 corner nodes.

    Node numbering (0-based corners): [0,1,2,3] typically in a loop.

      face=0 => [0, 1]
      face=1 => [1, 2]
      face=2 => [2, 3]
      face=3 => [3, 0]

    Parameters
    ----------
    face : int
        0-based edge index (0..3).

    Returns
    -------
    list of int
        The node indices on that edge.
    """
    faces_map = {
        0: [0, 1],
        1: [1, 2],
        2: [2, 3],
        3: [3, 0],
    }
    if face not in faces_map:
        raise ValueError("D2_nn4_quad has faces=0..3 only.")
    return faces_map[face]


def get_face_nodes_D2_nn8_quad(face: int):
    """
    Returns the 0-based node indices for a 2D 8-node quadrilateral (quadratic).
    The quad has 4 edges, face=0..3. Each edge has 2 corner nodes + 1 mid-edge node.

    Node numbering convention (0-based):
      corners: 0,1,2,3
      mid-edges: 4,5,6,7
    So
      face=0 => [0, 1, 4]
      face=1 => [1, 2, 5]
      face=2 => [2, 3, 6]
      face=3 => [3, 0, 7]

    Parameters
    ----------
    face : int
        0-based edge index (0..3).

    Returns
    -------
    list of int
        The node indices on that edge.
    """
    faces_map = {
        0: [0, 1, 4],
        1: [1, 2, 5],
        2: [2, 3, 6],
        3: [3, 0, 7],
    }
    if face not in faces_map:
        raise ValueError("D2_nn8_quad has faces=0..3 only.")
    return faces_map[face]
