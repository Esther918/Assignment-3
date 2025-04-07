import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from finiteelementanalysis import discretization as di
from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import discretization_demo_helper_fcns as di_demo


def plot_mesh_2D(fname: str, ele_type: str, coords: np.ndarray, connect: np.ndarray, gauss_points: np.ndarray = None):
    """
    Plots a 2D finite element mesh with aesthetically pleasing visualization.
    
    Parameters
    ----------
    fname : str
        The filename for saving the plot.
    ele_type : str
        The type of finite element.
    coords : np.ndarray of shape (n_nodes, 2)
        The coordinates of the nodes in physical space.
    connect : np.ndarray of shape (n_elements, n_nodes_per_element)
        The element connectivity matrix, specifying node indices for each element.
    gauss_points : np.ndarray of shape (num_elements, num_gauss_pts, 2), optional
        The physical coordinates of Gauss points for visualization.
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

    for elem_idx, element in enumerate(connect):
        element_coords = coords[element]

        if ele_type == "D2_nn3_tri":
            edges = [[0, 1], [1, 2], [2, 0]]
        elif ele_type == "D2_nn6_tri":
            edges = [[0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]]
        elif ele_type == "D2_nn4_quad":
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        elif ele_type == "D2_nn8_quad":
            edges = [[0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 3], [3, 7], [7, 0]]
        else:
            raise ValueError(f"Unsupported element type: {ele_type}")

        # Draw element edges in a soft gray
        for edge in edges:
            ax.plot(element_coords[edge, 0], element_coords[edge, 1], color='gray', lw=0.8, alpha=0.7)

        # Compute element center for labeling
        centroid = np.mean(element_coords, axis=0)
        ax.text(centroid[0], centroid[1], str(elem_idx), fontsize=9, ha='center', va='center', color='black', weight='bold')

    # Plot nodes
    ax.scatter(coords[:, 0], coords[:, 1], color=(0.8, 0.8, 0.8), s=20, edgecolors='red', linewidth=0.5, zorder=3)

    # Label nodes inside the circles in black
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=6, ha='center', va='center', color='black', weight='bold')

    # Plot Gauss points if provided
    if gauss_points is not None:
        gauss_x = gauss_points[:, :, 0].flatten()  # Extract all x-coordinates
        gauss_y = gauss_points[:, :, 1].flatten()  # Extract all y-coordinates
        ax.scatter(gauss_x, gauss_y, color='#AFCBFF', marker='o', s=15, edgecolors='black', linewidth=0.3, zorder=2, alpha=0.8, label="Gauss Points")

    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"2D Mesh Plot for {ele_type}", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(False)  # Remove grid for a cleaner look

    # Add legend if Gauss points are plotted
    if gauss_points is not None:
        ax.legend(loc='upper right', fontsize=10, frameon=False)

    # Save with high DPI and clean style
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return


def get_all_mesh_gauss_pts(ele_type: str, num_gauss_pts: int, coords: np.ndarray, connect: np.ndarray):
    """
    Computes the locations of Gauss points in physical coordinates for all elements in a 2D finite element mesh.

    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    num_gauss_pts : int
        The number of Gauss points to use for numerical integration in each element.
    coords : np.ndarray of shape (n_nodes, 2)
        The physical coordinates of the nodes in the mesh.
    connect : np.ndarray of shape (n_elements, n_nodes_per_element)
        The element connectivity matrix, specifying node indices for each element.

    Returns
    -------
    mesh_gauss_pts : np.ndarray of shape (num_elements, num_gauss_pts, 2)
        The computed Gauss point locations in physical coordinates for all elements.
    
    Notes
    -----
    - Uses shape function evaluations to transform Gauss points from reference coordinates 
      to physical coordinates.
    - Supports different element types with appropriate shape functions.
    """

    # Dictionary mapping element types to shape functions
    shape_fcn_dict = {
        "D2_nn3_tri": di.D2_nn3_tri,  # Linear triangle
        "D2_nn6_tri": di.D2_nn6_tri,  # Quadratic triangle
        "D2_nn4_quad": di.D2_nn4_quad,  # Bilinear quadrilateral
        "D2_nn8_quad": di.D2_nn8_quad   # Quadratic quadrilateral
    }

    if ele_type not in shape_fcn_dict:
        raise ValueError(f"Unsupported element type: {ele_type}")

    shape_fcn = shape_fcn_dict[ele_type]

    # Get Gauss points in natural coordinates
    gauss_pts, _ = di_demo.gauss_pts_and_weights(ele_type, num_gauss_pts)

    num_elements = connect.shape[0]
    _, _, num_nodes = di.element_info(ele_type)  # Get number of nodes per element
    dim = 2  # 2D problem

    # Initialize the result array
    mesh_gauss_pts = np.zeros((num_elements, num_gauss_pts, dim))

    # Compute the Gauss point locations in physical coordinates
    for kk in range(num_elements):
        # Get the coordinates of the element's nodes
        element_coords = np.array([coords[connect[kk, jj], :] for jj in range(num_nodes)])

        for jj in range(num_gauss_pts):
            # Evaluate the shape function at the Gauss point
            shape_fcn_eval = shape_fcn(gauss_pts[:, jj])

            # Compute physical coordinates using shape function interpolation
            physical_coords = element_coords.T @ shape_fcn_eval  # (2, num_nodes) @ (num_nodes,) → (2,)

            mesh_gauss_pts[kk, jj, :] = physical_coords.reshape((1, 2))  # Store result

    return mesh_gauss_pts


def interpolate_scalar_to_gauss_pts(ele_type: str, num_gauss_pts: int, fcn_to_interp, coords: np.ndarray, connect: np.ndarray):
    """
    Interpolates a given scalar function to Gauss points for all elements in a 2D finite element mesh.

    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    num_gauss_pts : int
        The number of Gauss points to use for numerical integration in each element.
    fcn_to_interp : callable
        A scalar function f(x, y) to interpolate at Gauss points.
    coords : np.ndarray of shape (n_nodes, 2)
        The physical coordinates of the nodes in the mesh.
    connect : np.ndarray of shape (n_elements, n_nodes_per_element)
        The element connectivity matrix, specifying node indices for each element.

    Returns
    -------
    fcn_interp_mesh_gauss_pts : np.ndarray of shape (num_elements, num_gauss_pts)
        Interpolated function values at each Gauss point in each element.

    Notes
    -----
    - Computes Gauss points in natural coordinates.
    - Evaluates function values at nodal positions.
    - Uses shape functions to interpolate function values at Gauss points.
    """

    # Dictionary mapping element types to shape functions
    shape_fcn_dict = {
        "D2_nn3_tri": di.D2_nn3_tri,  # Linear triangle
        "D2_nn6_tri": di.D2_nn6_tri,  # Quadratic triangle
        "D2_nn4_quad": di.D2_nn4_quad,  # Bilinear quadrilateral
        "D2_nn8_quad": di.D2_nn8_quad   # Quadratic quadrilateral
    }

    if ele_type not in shape_fcn_dict:
        raise ValueError(f"Unsupported element type: {ele_type}")

    shape_fcn = shape_fcn_dict[ele_type]

    # Get Gauss points in natural coordinates
    gauss_pts, _ = di_demo.gauss_pts_and_weights(ele_type, num_gauss_pts)

    num_elements = connect.shape[0]
    _, _, num_nodes = di.element_info(ele_type)  # Get number of nodes per element

    # Initialize the result array
    fcn_interp_mesh_gauss_pts = np.zeros((num_elements, num_gauss_pts))

    # Compute interpolated function values at Gauss points
    for kk in range(num_elements):
        # Get the coordinates of the element's nodes
        element_coords = np.array([coords[connect[kk, ii], :] for ii in range(num_nodes)])

        # Evaluate function at element nodes
        fcn_values_at_nodes = np.array([fcn_to_interp(x, y) for x, y in element_coords])

        for jj in range(num_gauss_pts):
            # Evaluate shape functions at the current Gauss point
            shape_fcn_eval = shape_fcn(gauss_pts[:, jj])

            # Interpolate function at Gauss points
            fcn_interp_mesh_gauss_pts[kk, jj] = (fcn_values_at_nodes.T @ shape_fcn_eval).item() # Dot product

    return fcn_interp_mesh_gauss_pts


def interpolate_scalar_deriv_to_gauss_pts(ele_type: str, num_gauss_pts: int, fcn_to_interp, coords: np.ndarray, connect: np.ndarray):
    """
    Interpolates a given scalar function derivative to Gauss points for all elements in a 2D finite element mesh.

    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    num_gauss_pts : int
        The number of Gauss points to use for numerical integration in each element.
    fcn_to_interp : callable
        A scalar function f(x, y) to interpolate at Gauss points.
    coords : np.ndarray of shape (n_nodes, 2)
        The physical coordinates of the nodes in the mesh.
    connect : np.ndarray of shape (n_elements, n_nodes_per_element)
        The element connectivity matrix, specifying node indices for each element.

    Returns
    -------
    fcn_interp_mesh_gauss_pts : np.ndarray of shape (num_elements, num_gauss_pts, 2)
        Interpolated function derivative values at each Gauss point in each element.

    Notes
    -----
    - Computes Gauss points in natural coordinates.
    - Evaluates function values at nodal positions.
    - Uses shape functions to interpolate function values at Gauss points.
    """

    # Dictionary mapping element types to shape functions
    shape_fcn_deriv_dict = {
        "D2_nn3_tri": di.D2_nn3_tri_dxi,  # Linear triangle
        "D2_nn6_tri": di.D2_nn6_tri_dxi,  # Quadratic triangle
        "D2_nn4_quad": di.D2_nn4_quad_dxi,  # Bilinear quadrilateral
        "D2_nn8_quad": di.D2_nn8_quad_dxi   # Quadratic quadrilateral
    }

    if ele_type not in shape_fcn_deriv_dict:
        raise ValueError(f"Unsupported element type: {ele_type}")

    shape_fcn_deriv = shape_fcn_deriv_dict[ele_type]

    # Get Gauss points in natural coordinates
    gauss_pts, _ = di_demo.gauss_pts_and_weights(ele_type, num_gauss_pts)

    num_elements = connect.shape[0]
    _, _, num_nodes = di.element_info(ele_type)  # Get number of nodes per element

    # Initialize the result array
    fcn_interp_mesh_gauss_pts = np.zeros((num_elements, num_gauss_pts, 2))

    # Compute interpolated function values at Gauss points
    for kk in range(num_elements):
        # Get the coordinates of the element's nodes
        element_coords = np.array([coords[connect[kk, ii], :] for ii in range(num_nodes)])

        # Evaluate function at element nodes
        fcn_values_at_nodes = np.array([fcn_to_interp(x, y) for x, y in element_coords])

        for jj in range(num_gauss_pts):
            # Evaluate shape functions at the current Gauss point
            dN_dxi = shape_fcn_deriv(gauss_pts[:, jj])
            J = element_coords.T @ dN_dxi  # (2, 2)
            J_inv = np.linalg.inv(J)

            # Interpolate function at Gauss points with Chain rule
            gradient_natural = dN_dxi.T @ fcn_values_at_nodes
            deriv_vec = J_inv.T @ gradient_natural
            fcn_interp_mesh_gauss_pts[kk, jj, 0] = deriv_vec[0]
            fcn_interp_mesh_gauss_pts[kk, jj, 1] = deriv_vec[1]

    return fcn_interp_mesh_gauss_pts


def plot_interpolation_with_error(
    fname: str,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    gauss_points_coords: np.ndarray,
    gauss_points_interp: np.ndarray,
    gauss_points_ground_truth: np.ndarray
):
    """
    Creates a side-by-side plot showing interpolated values at Gauss points
    and the corresponding absolute error compared to ground truth.

    Parameters
    ----------
    fname : str
        Filename to save the plot.
    ele_type : str
        The type of finite element (e.g., "D2_nn6_tri").
    coords : np.ndarray
        Node coordinates (n_nodes, 2).
    connect : np.ndarray
        Connectivity matrix (n_elements, nodes_per_element).
    gauss_points_coords : np.ndarray
        Coordinates of Gauss points (n_elements, num_gauss_pts, 2).
    gauss_points_interp : np.ndarray
        Interpolated values at Gauss points (n_elements, num_gauss_pts).
    gauss_points_ground_truth : np.ndarray
        Ground truth values at Gauss points (n_elements, num_gauss_pts).
    """
    # Define edge connectivity by element type
    edge_map = {
        "D2_nn3_tri": [[0, 1], [1, 2], [2, 0]],
        "D2_nn6_tri": [[0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]],
        "D2_nn4_quad": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "D2_nn8_quad": [[0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 3], [3, 7], [7, 0]]
    }

    if ele_type not in edge_map:
        raise ValueError(f"Unsupported element type: {ele_type}")
    edges = edge_map[ele_type]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    # Flatten Gauss point data for plotting
    gauss_x = gauss_points_coords[:, :, 0].flatten()
    gauss_y = gauss_points_coords[:, :, 1].flatten()
    interp_vals = gauss_points_interp.flatten()
    error_vals = np.abs(gauss_points_interp - gauss_points_ground_truth).flatten()

    # Plot Interpolated and Error values
    for ax_idx, (ax, data, title) in enumerate(zip(
        axes,
        [interp_vals, error_vals],
        ["Interpolated Values at Gauss Points", "Absolute Interpolation Error, max= %0.6e" % (np.max(error_vals))]
    )):
        # Draw element edges
        for element in connect:
            element_coords = coords[element]
            for edge in edges:
                try:
                    ax.plot(
                        element_coords[edge, 0], element_coords[edge, 1],
                        color='gray', lw=0.8, alpha=0.7
                    )
                except IndexError:
                    continue  # Safely skip malformed edge refs

        # Draw mesh nodes
        ax.scatter(coords[:, 0], coords[:, 1], s=8, color='black', zorder=2)

        # Plot Gauss points with scalar value
        gauss_pt_size = 5 + int(100 / connect.shape[0])
        sc = ax.scatter(
            gauss_x, gauss_y, c=data, cmap='coolwarm',
            edgecolor='k', linewidth=0.2, s=gauss_pt_size, zorder=3
        )
        fig.colorbar(sc, ax=ax, label="Value" if ax_idx == 0 else "Error")

        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_aspect('equal')

    plt.suptitle(f"Gauss Point Interpolation for {ele_type}", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return


def plot_interpolation_gradient_with_error(
    fname: str,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    gauss_points_coords: np.ndarray,
    gauss_points_interp: np.ndarray,
    gauss_points_ground_truth: np.ndarray
):
    """
    Creates a 2x2 subplot grid showing interpolated gradients (x and y components)
    and their absolute errors compared to ground truth at Gauss points.

    Parameters
    ----------
    fname : str
        Filename to save the plot.
    ele_type : str
        The type of finite element (e.g., "D2_nn6_tri").
    coords : np.ndarray
        Node coordinates (n_nodes, 2).
    connect : np.ndarray
        Connectivity matrix (n_elements, nodes_per_element).
    gauss_points_coords : np.ndarray
        Coordinates of Gauss points (n_elements, num_gauss_pts, 2).
    gauss_points_interp : np.ndarray
        Interpolated gradients at Gauss points (n_elements, num_gauss_pts, 2).
    gauss_points_ground_truth : np.ndarray
        Ground truth gradients at Gauss points (n_elements, num_gauss_pts, 2).
    """
    edge_map = {
        "D2_nn3_tri": [[0, 1], [1, 2], [2, 0]],
        "D2_nn6_tri": [[0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]],
        "D2_nn4_quad": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "D2_nn8_quad": [[0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 3], [3, 7], [7, 0]]
    }

    if ele_type not in edge_map:
        raise ValueError(f"Unsupported element type: {ele_type}")
    edges = edge_map[ele_type]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

    # Flatten Gauss point coordinates
    gauss_x = gauss_points_coords[:, :, 0].flatten()
    gauss_y = gauss_points_coords[:, :, 1].flatten()

    # For each component (x-derivative, y-derivative)
    for i, comp in enumerate(['x', 'y']):
        interp_vals = gauss_points_interp[:, :, i].flatten()
        ground_truth = gauss_points_ground_truth[:, :, i].flatten()
        error_vals = np.abs(interp_vals - ground_truth)

        for j, (data, title) in enumerate([
            (interp_vals, f"Interpolated ∂f/∂{comp}"),
            (error_vals, f"Absolute Error in ∂f/∂{comp}, max= {np.max(error_vals):.2e}")
        ]):
            ax = axes[i][j]

            # Draw element edges
            for element in connect:
                element_coords = coords[element]
                for edge in edges:
                    try:
                        ax.plot(
                            element_coords[edge, 0], element_coords[edge, 1],
                            color='gray', lw=0.8, alpha=0.7
                        )
                    except IndexError:
                        continue  # skip malformed edge indices

            # Draw mesh nodes
            ax.scatter(coords[:, 0], coords[:, 1], s=8, color='black', zorder=2)

            # Plot Gauss point data
            gauss_pt_size = 5 + int(100 / connect.shape[0])
            sc = ax.scatter(
                gauss_x, gauss_y, c=data, cmap='coolwarm',
                edgecolor='k', linewidth=0.2, s=gauss_pt_size, zorder=3
            )
            fig.colorbar(sc, ax=ax, label="Value" if j == 0 else "Error")

            ax.set_title(title, fontsize=12, weight='bold')
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_aspect('equal')

    plt.suptitle(f"Gauss Point Gradient Interpolation for {ele_type}", fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return


def compute_element_quality_metrics(ele_type, coords, connect):
    """
    Computes quality metrics for each element in a 2D finite element mesh.

    Parameters
    ----------
    ele_type : str
        The type of finite element ("D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad").
    coords : np.ndarray
        Node coordinates (n_nodes, 2).
    connect : np.ndarray
        Connectivity matrix (n_elements, nodes_per_element).

    Returns
    -------
    aspect_ratios : np.ndarray
        Aspect ratio for each element.
    skewness : np.ndarray
        Skewness for each element.
    min_angles : np.ndarray
        Minimum interior angle for each element.
    max_angles : np.ndarray
        Maximum interior angle for each element.
    """
    num_elements = connect.shape[0]

    # Define edges for aspect ratio (only corner nodes)
    edge_map = {
        "D2_nn3_tri": [[0, 1], [1, 2], [2, 0]],
        "D2_nn6_tri": [[0, 1], [1, 2], [2, 0]],  # Use only corners
        "D2_nn4_quad": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "D2_nn8_quad": [[0, 1], [1, 2], [2, 3], [3, 0]]  # Corners only
    }

    if ele_type not in edge_map:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    edges = edge_map[ele_type]

    aspect_ratios = np.zeros(num_elements)
    skewness = np.zeros(num_elements)
    min_angles = np.zeros(num_elements)
    max_angles = np.zeros(num_elements)

    for i in range(num_elements):
        element_nodes = coords[connect[i]]

        # --- Aspect Ratio ---
        if ele_type in ["D2_nn3_tri", "D2_nn6_tri"]:
            corner_nodes = element_nodes[:3]
            A, B, C = corner_nodes
            a = np.linalg.norm(B - C)
            b = np.linalg.norm(C - A)
            c = np.linalg.norm(A - B)

            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
            l_min = min(a, b, c)

            if area > 0 and l_min > 0:
                R = (a * b * c) / (4 * area)
                aspect_ratios[i] = R / l_min
            else:
                aspect_ratios[i] = np.inf  # Degenerate triangle
        else:
            edge_lengths = np.array([
                np.linalg.norm(element_nodes[e[1]] - element_nodes[e[0]]) for e in edges
            ])
            l_max = edge_lengths.max()
            l_min = edge_lengths.min()
            aspect_ratios[i] = l_max / l_min if l_min > 0 else np.inf

        # --- Skewness (centroid-based) ---
        centroid = np.mean(element_nodes, axis=0)
        distances = np.linalg.norm(element_nodes - centroid, axis=1)
        d_max = np.max(distances)
        d_min = np.min(distances)
        skewness[i] = (d_max - d_min) / d_max if d_max > 0 else 0.0

        # --- Angles (based on corner nodes only) ---
        if "tri" in ele_type.lower():
            nodes_to_use = element_nodes[:3]
        elif "quad" in ele_type.lower():
            nodes_to_use = element_nodes[:4]
        else:
            continue  # skip unsupported elements

        angles = []
        n = len(nodes_to_use)
        for j in range(n):
            a = nodes_to_use[j - 1]
            b = nodes_to_use[j]
            c = nodes_to_use[(j + 1) % n]
            ba = a - b
            bc = c - b
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            if norm_ba > 0 and norm_bc > 0:
                cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
                angle = np.arccos(np.clip(cosine, -1.0, 1.0))
                angles.append(np.degrees(angle))

        if angles:
            min_angles[i] = min(angles)
            max_angles[i] = max(angles)
        else:
            min_angles[i] = 0
            max_angles[i] = 180

    return aspect_ratios, skewness, min_angles, max_angles


def compute_condition_and_jacobian(ele_type, coords, connect):
    """
    Computes the condition number and Jacobian determinant at the element center (reference centroid)
    for each element in a 2D finite element mesh.

    Parameters
    ----------
    ele_type : str
        Element type. Supported values:
        - "D2_nn3_tri" : 3-node triangle
        - "D2_nn6_tri" : 6-node triangle
        - "D2_nn4_quad" : 4-node quadrilateral
        - "D2_nn8_quad" : 8-node quadrilateral
    coords : np.ndarray
        Node coordinates of the mesh, shape (n_nodes, 2)
    connect : np.ndarray
        Element connectivity, shape (n_elements, n_nodes_per_elem)

    Returns
    -------
    condition_numbers : np.ndarray of shape (n_elements,)
        The condition number of the Jacobian matrix at the element center for each element.
    jacobian_dets : np.ndarray of shape (n_elements,)
        The determinant of the Jacobian matrix at the element center for each element.
    """
    # Map element type to shape function derivative function
    shape_fcn_derivs_dict = {
        "D2_nn3_tri": di.D2_nn3_tri_dxi,  # Linear triangle
        "D2_nn6_tri": di.D2_nn6_tri_dxi,  # Quadratic triangle
        "D2_nn4_quad": di.D2_nn4_quad_dxi,  # Bilinear quadrilateral
        "D2_nn8_quad": di.D2_nn8_quad_dxi   # Quadratic quadrilateral
    }

    if ele_type not in shape_fcn_derivs_dict:
        raise ValueError(f"Unsupported element type: {ele_type}")

    shape_fcn_derivs = shape_fcn_derivs_dict[ele_type]

    # Define the reference element center
    if ele_type in ["D2_nn3_tri", "D2_nn6_tri"]:
        sample_pts = np.array([[1/3, 1/3]])  # Barycentric center
    elif ele_type in ["D2_nn4_quad", "D2_nn8_quad"]:
        sample_pts = np.array([[0.0, 0.0]])  # Center of reference square

    n_elements = connect.shape[0]

    condition_numbers = np.zeros(n_elements)
    jacobian_dets = np.zeros(n_elements)

    for e in range(n_elements):
        element_coords = coords[connect[e]]  # shape (n_nodes_per_elem, 2)

        xi_eta = sample_pts[0]  # Only one point (the center)
        dN_dxi = shape_fcn_derivs(xi_eta)  # shape (n_nodes_per_elem, 2)

        # Compute Jacobian: J = dN_dxi.T @ element_coords
        J = dN_dxi.T @ element_coords  # shape (2, 2)

        jacobian_dets[e] = np.linalg.det(J)
        condition_numbers[e] = np.linalg.cond(J)

    return condition_numbers, jacobian_dets


def plot_element_quality_histograms(
    fname: str,
    super_title: str,
    ele_type: str,
    cond_nums: np.ndarray,
    jac_dets: np.ndarray,
    aspect_ratios: np.ndarray,
    skewness: np.ndarray,
    min_angles: np.ndarray,
    max_angles: np.ndarray
):
    """
    Plots a 3x2 grid of histograms for element quality metrics and saves the figure.

    Parameters
    ----------
    fname : str
        Path to save the resulting figure (e.g., "test/files/quality_hist.png").
    super_title : str
        A string to display as the overall title across all subplots.
    ele_type : str
        Element type string (e.g., "D2_nn3_tri", "D2_nn4_quad")
    cond_nums : np.ndarray
        Condition numbers of the Jacobian matrix for each element.
    jac_dets : np.ndarray
        Determinants of the Jacobian matrix for each element.
    aspect_ratios : np.ndarray
        Aspect ratio for each element.
    skewness : np.ndarray
        Skewness for each element.
    min_angles : np.ndarray
        Minimum interior angle (in degrees) per element.
    max_angles : np.ndarray
        Maximum interior angle (in degrees) per element.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=300)
    fig.suptitle(super_title, fontsize=16, fontweight='bold')

    # Define each metric and corresponding titles
    data = [
        (aspect_ratios, "Aspect Ratio", "lightblue"),
        (skewness, "Skewness", "lightgray"),
        (cond_nums, "Condition Number", "lightblue"),
        (jac_dets, "Jacobian Determinant", "lightgray"),
        (min_angles, "Minimum Angle (°)", "lightblue"),
        (max_angles, "Maximum Angle (°)", "lightgray"),
    ]

    # Triangle vs quad logic
    is_triangle = "tri" in ele_type.lower()

    for ax, (values, title, color) in zip(axes.flatten(), data):
        # Use log scale for Condition Number
        if "Condition Number" in title:
            ax.hist(values, bins=30, color=color, edgecolor='black', linewidth=0.5)
            ax.set_xscale("log")
        else:
            ax.hist(values, bins=30, color=color, edgecolor='black', linewidth=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylabel("Count")

        # Red guideline overlays
        if "Aspect Ratio" in title:
            if is_triangle:
                ax.axvline(2, color='red', linestyle='--', label='Suggested Upper Limit')
            else:
                ax.axvline(5, color='red', linestyle='--', label='Suggested Upper Limit')
            ax.legend(fontsize=8)
        elif "Skewness" in title:
            threshold = 0.8 if is_triangle else 2.0
            ax.axvline(threshold, color='red', linestyle='--', label=f'Suggested Upper Limit')
            ax.legend(fontsize=8)
        elif "Condition Number" in title:
            ax.axvline(1000, color='red', linestyle='--', label='Suggested Upper Limit')
            ax.legend(fontsize=8)
        elif "Jacobian Determinant" in title:
            ax.axvline(0, color='red', linestyle='--', label='Min Acceptable > 0')
            ax.legend(fontsize=8)
        elif "Minimum Angle" in title:
            ax.axvline(30, color='red', linestyle='--', label='Suggested Lower Limit')
            ax.legend(fontsize=8)
        elif "Maximum Angle" in title:
            if is_triangle:
                ax.axvline(120, color='red', linestyle='--', label='Suggested Upper Limit')
            else:
                ax.axvline(150, color='red', linestyle='--', label='Suggested Upper Limit')
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return

