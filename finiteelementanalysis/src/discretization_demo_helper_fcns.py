import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from finiteelementanalysis import discretization as di
import numpy as np


def interpolate_field_natural_coords_single_element(ele_type, node_values, xi_vals, eta_vals):
    """
    Interpolates a scalar field inside a single finite element using its shape functions
    in natural (reference) coordinates (ξ, η).

    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    node_values : numpy.ndarray of shape (n_nodes,)
        The values of the field at the element nodes.
    xi_vals : numpy.ndarray of shape (n_xi,)
        The natural coordinate values (ξ) at which interpolation is performed.
    eta_vals : numpy.ndarray of shape (n_eta,)
        The natural coordinate values (η) at which interpolation is performed.

    Returns
    -------
    interpolated_vals : numpy.ndarray of shape (n_xi, n_eta)
        The interpolated field values at the specified (ξ, η) points.

    Raises
    ------
    ValueError
        If an unsupported element type is provided.

    Notes
    -----
    - This function assumes that the element is in **natural coordinates** (ξ, η).
    - The function selects the appropriate shape function for the given element type.
    - Shape functions are evaluated at the given (ξ, η) values to interpolate the field.
    - Supports both triangular and quadrilateral elements.
    """
    shape_function_map = {
        "D2_nn3_tri": di.D2_nn3_tri,
        "D2_nn6_tri": di.D2_nn6_tri,
        "D2_nn4_quad": di.D2_nn4_quad,
        "D2_nn8_quad": di.D2_nn8_quad,
    }

    if ele_type not in shape_function_map:
        raise ValueError(f"Unsupported element type: {ele_type}")

    shape_function = shape_function_map[ele_type]

    interpolated_vals = np.zeros((len(xi_vals)))
    for i in range(0, len(xi_vals)):
        xi = xi_vals[i]
        eta = eta_vals[i]
        N = shape_function(np.asarray([xi, eta]))
        interpolated_vals[i] = np.dot(N.T, node_values).item()

    return interpolated_vals.reshape((-1, 1))


def plot_interpolate_field_natural_coords_single_element(fname: str, ele_type: str, node_values: np.ndarray, num_interp_pts: int = 10):
    """
    Plots a scalar field interpolated across a sampling of points in natural coordinates.
    Saves the file according to `fname`.
    Calls `interpolate_field_natural_coords_single_element` to perform interpolation.
    
    Parameters
    ----------
    fname : str
        The filename to save the plot.
    ele_type : str
        The type of finite element.
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    node_values : numpy.ndarray of shape (n_nodes,)
        The values of the field at the element nodes.
    num_interp_pts : int, optional
        The number of interpolation points along each axis (default is 10).
    """
    # Define sampling points in natural coordinates
    if ele_type in ["D2_nn3_tri", "D2_nn6_tri"]:
        xi_vals = np.linspace(0, 1, num_interp_pts)
        eta_vals = np.linspace(0, 1, num_interp_pts)
        XI, ETA = np.meshgrid(xi_vals, eta_vals)
        mask = XI + ETA <= 1  # Valid points inside the triangle
        xi_filtered = XI[mask].flatten()
        eta_filtered = ETA[mask].flatten()
        ref_nodes = np.array([[1, 0], [0, 1], [0, 0]]) if ele_type == "D2_nn3_tri" else np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0]])
    
    elif ele_type in ["D2_nn4_quad", "D2_nn8_quad"]:
        xi_vals = np.linspace(-1, 1, num_interp_pts)
        eta_vals = np.linspace(-1, 1, num_interp_pts)
        XI, ETA = np.meshgrid(xi_vals, eta_vals)
        xi_filtered = XI.flatten()
        eta_filtered = ETA.flatten()
        ref_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) if ele_type == "D2_nn4_quad" else np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    else:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    # Compute interpolated field values
    interpolated_vals = interpolate_field_natural_coords_single_element(ele_type, node_values, xi_filtered, eta_filtered)
    
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(xi_filtered, eta_filtered, c=interpolated_vals.flatten(), cmap='coolwarm', edgecolors='k', s=50, alpha=0.8)
    plt.colorbar(label='Interpolated Field')
    
    # Plot element boundaries
    if ele_type in ["D2_nn3_tri", "D2_nn6_tri"]:
        tri_nodes = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])  # Reference triangle
        plt.plot(tri_nodes[:, 0], tri_nodes[:, 1], 'k-', lw=2)
    elif ele_type in ["D2_nn4_quad", "D2_nn8_quad"]:
        quad_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])  # Reference quad
        plt.plot(quad_nodes[:, 0], quad_nodes[:, 1], 'k-', lw=2)
    
    # Label reference element nodes in natural coordinates
    for i, (xi, eta) in enumerate(ref_nodes):
        plt.text(xi, eta, f'N{i+1}', fontsize=10, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.6))
    
    plt.xlabel("ξ (Natural Coordinate)")
    plt.ylabel("η (Natural Coordinate)")
    plt.title(f"Interpolated Field for {ele_type}")
    plt.savefig(fname, dpi=300)
    plt.close()
    return


def visualize_isoparametric_mapping_single_element(fname: str, ele_type, node_coords, node_values, num_interp_pts=20):
    """
    Visualizes the isoparametric mapping of a reference element to its physical shape.
    Calls `interpolate_field_natural_coords_single_element` to interpolate values inside the element.
    
    Parameters
    ----------
    fname : str
        The filename to save the plot.
    ele_type : str
        The type of finite element.
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    node_coords : numpy.ndarray of shape (n_nodes, 2)
        The physical coordinates of the element nodes.
    node_values : numpy.ndarray of shape (n_nodes,)
        The values of the field at the element nodes.
    num_interp_pts : int, optional
        The number of interpolation points along each axis (default is 20 for smoother results).
    """
    # Define sampling points in natural coordinates
    if ele_type in ["D2_nn3_tri", "D2_nn6_tri"]:
        xi_vals = np.linspace(0, 1, num_interp_pts)
        eta_vals = np.linspace(0, 1, num_interp_pts)
        XI, ETA = np.meshgrid(xi_vals, eta_vals)
        mask = XI + ETA <= 1  # Filter points inside the reference triangle
        xi_filtered = XI[mask]
        eta_filtered = ETA[mask]
        ref_nodes = np.array([[1, 0], [0, 1], [0, 0]]) if ele_type == "D2_nn3_tri" else np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0]])

    elif ele_type in ["D2_nn4_quad", "D2_nn8_quad"]:
        xi_vals = np.linspace(-1, 1, num_interp_pts)
        eta_vals = np.linspace(-1, 1, num_interp_pts)
        XI, ETA = np.meshgrid(xi_vals, eta_vals)
        xi_filtered = XI.flatten()
        eta_filtered = ETA.flatten()
        ref_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) if ele_type == "D2_nn4_quad" else np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    else:
        raise ValueError(f"Unsupported element type: {ele_type}")

    # Compute interpolated field values and mapped physical coordinates
    interpolated_vals = interpolate_field_natural_coords_single_element(ele_type, node_values, xi_filtered, eta_filtered).flatten()
    x_mapped = interpolate_field_natural_coords_single_element(ele_type, node_coords[:, 0], xi_filtered, eta_filtered).flatten()
    y_mapped = interpolate_field_natural_coords_single_element(ele_type, node_coords[:, 1], xi_filtered, eta_filtered).flatten()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot natural coordinates reference element
    sc1 = axs[0].scatter(xi_filtered, eta_filtered, c=interpolated_vals, cmap='coolwarm', edgecolors='k', s=50, alpha=0.8)
    axs[0].set_xlabel("ξ (Natural Coordinate)")
    axs[0].set_ylabel("η (Natural Coordinate)")
    axs[0].set_title("Reference Element (Natural Coordinates)")
    axs[0].set_aspect('equal')
    fig.colorbar(sc1, ax=axs[0], label='Interpolated Field')

    # Label reference element nodes in natural coordinates
    for i, (xi, eta) in enumerate(ref_nodes):
        axs[0].text(xi, eta, f'N{i+1}', fontsize=10, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.6))

    # Plot physical coordinates mapped element
    point_sizes = 40 * (1 + np.abs(x_mapped - np.mean(x_mapped)))  # Adjust size based on mapping distortion
    sc2 = axs[1].scatter(x_mapped, y_mapped, c=interpolated_vals, cmap='coolwarm', edgecolors='k', s=point_sizes, alpha=0.8)
    axs[1].set_xlabel("x (Physical Coordinate)")
    axs[1].set_ylabel("y (Physical Coordinate)")
    axs[1].set_title("Mapped Element (Physical Coordinates)")
    axs[1].set_aspect('equal')
    fig.colorbar(sc2, ax=axs[1], label='Interpolated Field')

    # Label element nodes in physical space
    for i, (x, y) in enumerate(node_coords):
        axs[1].text(x, y, f'N{i+1}', fontsize=10, ha='center', va='center', color='white', bbox=dict(facecolor='black', alpha=0.6))

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return


def compute_jacobian(ele_type, node_coords, xi, eta):
    """
    Computes the Jacobian matrix for a given element type at a specified natural coordinate (ξ, η).
    
    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    node_coords : numpy.ndarray of shape (n_nodes, 2)
        The physical coordinates of the element nodes.
    xi : float
        The ξ natural coordinate where the Jacobian is evaluated.
    eta : float
        The η natural coordinate where the Jacobian is evaluated.
    
    Returns
    -------
    J : numpy.ndarray of shape (2,2)
        The Jacobian matrix at (ξ, η).
    """
    # Define mapping between element type and derivative functions
    shape_function_derivatives = {
        "D2_nn3_tri": di.D2_nn3_tri_dxi,
        "D2_nn6_tri": di.D2_nn6_tri_dxi,
        "D2_nn4_quad": di.D2_nn4_quad_dxi,
        "D2_nn8_quad": di.D2_nn8_quad_dxi,
    }

    if ele_type not in shape_function_derivatives:
        raise ValueError(f"Unsupported element type: {ele_type}")

    # Compute the shape function derivatives in natural coordinates
    dN_dxi = shape_function_derivatives[ele_type](np.array([xi, eta]))

    # Compute the Jacobian matrix using matrix multiplication
    J = node_coords.T @ dN_dxi

    return J


def interpolate_gradient_natural_coords_single_element(ele_type, node_values, xi_vals, eta_vals):
    """
    Interpolates the gradient of a scalar or vector field in natural coordinates (ξ, η).
    
    Parameters
    ----------
    ele_type : str
        The type of finite element.
    node_values : numpy.ndarray of shape (n_nodes,) or (n_nodes, 2)
        The values of the field at the element nodes.
        - If shape is (n_nodes,), the function interpolates a scalar field.
        - If shape is (n_nodes, 2), the function interpolates a vector field (e.g., displacement or velocity).
    xi_vals : numpy.ndarray of shape (n_xi,)
        The natural coordinate values (ξ) at which interpolation is performed.
    eta_vals : numpy.ndarray of shape (n_eta,)
        The natural coordinate values (η) at which interpolation is performed.

    Returns
    -------
    gradient_natural : numpy.ndarray
        The interpolated field gradient in natural coordinates.
        - If interpolating a scalar field, the shape is (2, n_xi * n_eta).
        - If interpolating a vector field, the shape is (2, n_xi * n_eta, 2), where the last dimension corresponds
          to the two field components.
    """
    shape_function_derivatives = {
        "D2_nn3_tri": di.D2_nn3_tri_dxi,
        "D2_nn6_tri": di.D2_nn6_tri_dxi,
        "D2_nn4_quad": di.D2_nn4_quad_dxi,
        "D2_nn8_quad": di.D2_nn8_quad_dxi,
    }

    if ele_type not in shape_function_derivatives:
        raise ValueError(f"Unsupported element type: {ele_type}")

    # Determine if the input is scalar or vector field
    scalar_field = len(node_values.shape) == 1 or node_values.shape[1] == 1
    n_field_components = 1 if scalar_field else node_values.shape[1]
    gradient_natural = np.zeros((2, len(xi_vals) * len(eta_vals), n_field_components))

    index = 0
    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            dN_dxi = shape_function_derivatives[ele_type](np.array([xi, eta]))
            
            if scalar_field:
                gradient_natural[:, index] = (dN_dxi.T @ node_values).reshape((2, 1))
            else:
                gradient_natural[:, index, :] = (dN_dxi.T @ node_values).reshape((2, 2))

            index += 1

    return gradient_natural


def transform_gradient_to_physical(ele_type, node_coords, xi_vals, eta_vals, gradient_natural):
    """
    Transforms the interpolated gradient from natural coordinates (ξ, η) to physical coordinates (x, y).
    
    Parameters
    ----------
    ele_type : str
        The type of finite element.
    node_coords : numpy.ndarray of shape (n_nodes, 2)
        The physical coordinates of the element nodes.
    xi_vals : numpy.ndarray of shape (n_xi,)
        The natural coordinate values (ξ) at which transformation is performed.
    eta_vals : numpy.ndarray of shape (n_eta,)
        The natural coordinate values (η) at which transformation is performed.
    gradient_natural : numpy.ndarray of shape (2, n_xi * n_eta) or (2, n_xi * n_eta, n_components)
        The interpolated gradient in natural coordinates.
        - If interpolating a scalar field, shape is (2, n_xi * n_eta).
        - If interpolating a vector field, shape is (2, n_xi * n_eta, n_components).
    
    Returns
    -------
    gradient_physical : numpy.ndarray of shape (2, n_xi * n_eta) or (2, n_xi * n_eta, n_components)
        The transformed field gradient in physical coordinates.
    """
    # Determine if scalar or vector field
    is_scalar_field = gradient_natural.shape[2] == 1
    n_components = 1 if is_scalar_field else gradient_natural.shape[2]

    # Initialize transformed gradient
    gradient_physical = np.zeros_like(gradient_natural)

    index = 0
    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            J = compute_jacobian(ele_type, node_coords, xi, eta)
            J_inv = np.linalg.inv(J)

            if is_scalar_field:
                gradient_physical[:, index] = J_inv.T @ gradient_natural[:, index]
            else:
                for component in range(n_components):
                    gradient_physical[:, index, component] = J_inv.T @ gradient_natural[:, index, component]
                    
            index += 1

    return gradient_physical


def gauss_pts_and_weights(ele_type, num_pts):
    """
    Retrieves the Gauss quadrature points and weights for a given element type and number of integration points.
    
    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    num_pts : int
        The number of Gauss integration points.
        - Triangular elements: Supports 1, 3, or 4 points.
        - Quadrilateral elements: Supports 1, 4, or 9 points.
    
    Returns
    -------
    gauss_pts : np.ndarray of shape (2, num_pts)
        The Gauss quadrature points for the specified element type.
    gauss_weights : np.ndarray of shape (num_pts, 1)
        The corresponding Gauss quadrature weights.
    
    Raises
    ------
    ValueError
        If an unsupported element type is provided.
    """
    gauss_pts_all = {
        "D2_nn3_tri": di.gauss_points_2d_triangle,
        "D2_nn6_tri": di.gauss_points_2d_triangle,
        "D2_nn4_quad": di.gauss_points_2d_quad,
        "D2_nn8_quad": di.gauss_points_2d_quad,
    }

    gauss_weights_all = {
        "D2_nn3_tri": di.gauss_weights_2d_triangle,
        "D2_nn6_tri": di.gauss_weights_2d_triangle,
        "D2_nn4_quad": di.gauss_weights_2d_quad,
        "D2_nn8_quad": di.gauss_weights_2d_quad,
    }
    
    if ele_type not in gauss_pts_all:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    gauss_pts = gauss_pts_all[ele_type](num_pts)
    gauss_weights = gauss_weights_all[ele_type](num_pts)
    
    return gauss_pts, gauss_weights


def visualize_gauss_pts(fname, ele_type, num_pts):
    """
    Visualizes Gauss quadrature points and element nodes in natural coordinates.
    
    Parameters
    ----------
    fname : str
        The filename to save the plot.
    ele_type : str
        The type of finite element (e.g., "D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad").
    num_pts : int
        The number of Gauss integration points.
    
    Saves
    -----
    A figure displaying the element's reference shape with labeled nodes, mid-edge nodes (if applicable),
    and Gauss points.
    """
    # Get Gauss points
    gauss_pts, _ = gauss_pts_and_weights(ele_type, num_pts)
    
    # Define reference element nodes in natural coordinates
    if ele_type == "D2_nn3_tri":
        nodes = np.array([[1, 0], [0, 1], [0, 0]])  # Reference triangle
        edges = [[0, 1], [1, 2], [2, 0]]
    elif ele_type == "D2_nn6_tri":
        nodes = np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0]])  # Quadratic triangle
        edges = [[0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]]
    elif ele_type == "D2_nn4_quad":
        nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])  # Reference quadrilateral
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    elif ele_type == "D2_nn8_quad":
        nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])  # Quadratic quad
        edges = [[0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 3], [3, 7], [7, 0]]
    else:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot element edges
    for edge in edges:
        ax.plot(nodes[edge, 0], nodes[edge, 1], 'k-', lw=2)
    
    # Plot and label element nodes
    for i, (x, y) in enumerate(nodes):
        ax.scatter(x, y, color='blue', s=100, edgecolors='k', zorder=3)
        ax.text(x, y, f'N{i+1}', fontsize=12, ha='right', va='bottom', color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot and label Gauss points
    for i in range(gauss_pts.shape[1]):
        x, y = gauss_pts[:, i]
        ax.scatter(x, y, color='red', s=80, edgecolors='k', zorder=3)
        ax.text(x, y, f'G{i+1}', fontsize=12, ha='left', va='top', color='red', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel("ξ (Natural Coordinate)")
    ax.set_ylabel("η (Natural Coordinate)")
    ax.set_title(f"Gauss Points and Element Nodes for {ele_type}")
    ax.set_aspect('equal')
    
    plt.savefig(fname, dpi=300)
    plt.close()
    return


def compute_integral_of_derivative(ele_type, num_gauss_pts, node_coords, nodal_values):
    """
    Computes the integral of the derivative of a given quantity over a finite element using Gaussian quadrature.
    
    Parameters
    ----------
    ele_type : str
        The type of finite element.
    num_gauss_pts : int
        The number of Gauss integration points.
    node_coords : np.ndarray of shape (n_nodes, 2)
        The nodal coordinates of the element in physical space.
    nodal_values : np.ndarray of shape (n_nodes,) or (n_nodes, n_components)
        The values of the quantity at the element nodes. Can be scalar or vector-valued.
    
    Returns
    -------
    integral : np.ndarray
        The computed integral of the derivative over the element.
        - If `nodal_values` is scalar, the output is (2,).
        - If `nodal_values` is a vector field, the output is (2, n_components).
    """
    # Get Gauss points and weights
    gauss_pts, gauss_weights = gauss_pts_and_weights(ele_type, num_gauss_pts)

    is_vector_field = len(nodal_values.shape) == 2
    n_components = 1 if not is_vector_field else nodal_values.shape[1]
    integral = np.zeros((2, n_components)) if is_vector_field else np.zeros(2)

    for i in range(num_gauss_pts):
        xi, eta = gauss_pts[:, i]
        weight = gauss_weights[i, 0]

        # Compute gradient in natural coordinates
        gradient_natural = interpolate_gradient_natural_coords_single_element(
            ele_type, nodal_values, np.array([xi]), np.array([eta])
        )

        # Transform gradient to physical coordinates
        gradient_physical = transform_gradient_to_physical(
            ele_type, node_coords, np.array([xi]), np.array([eta]), gradient_natural
        )

        # Compute the determinate of the Jacobian
        J = compute_jacobian(ele_type, node_coords, xi, eta)
        det_J = np.linalg.det(J)

        # Compute integral by summing the weighted gradient contributions
        integral += weight * gradient_physical[:, 0, :] * det_J

    return integral


