import gmsh
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def generate_rect_mesh_2d(
    ele_type: str,
    x_lower: float,
    y_lower: float,
    x_upper: float,
    y_upper: float,
    nx: int,
    ny: int
):
    """
    Generate a 2D rectangular mesh for one of the following element types:
      - D2_nn3_tri   : 3-node linear triangles (tri3)
      - D2_nn6_tri   : 6-node quadratic triangles (tri6)
      - D2_nn4_quad  : 4-node bilinear quadrilaterals (quad4)
      - D2_nn8_quad  : 8-node quadratic quadrilaterals (quad8)

    The domain is [x_lower, x_upper] x [y_lower, y_upper]. The integer nx, ny
    specify how many element slices along x and y. For example:
      - If ele_type='D2_nn4_quad' and nx=3, ny=2, you get a 3 x 2 grid of quad4
        elements => total 3*2=6 elements.
      - If ele_type='D2_nn3_tri', each rectangular cell is split into 2 triangles,
        so total elements = 2 * nx * ny, and so on.

    Parameters
    ----------
    ele_type : str
        One of {'D2_nn3_tri', 'D2_nn6_tri', 'D2_nn4_quad', 'D2_nn8_quad'}.
    x_lower, y_lower : float
        Coordinates of the lower-left corner of the domain.
    x_upper, y_upper : float
        Coordinates of the upper-right corner of the domain.
    nx, ny : int
        Number of subdivisions (elements in each direction) along x and y.

    Returns
    -------
    coords : numpy.ndarray
        Node coordinates, shape (n_nodes, 2).
    connect : numpy.ndarray
        Element connectivity, shape depends on element type:
          - tri3  -> (n_elem, 3)
          - tri6  -> (n_elem, 6)
          - quad4 -> (n_elem, 4)
          - quad8 -> (n_elem, 8)

    Notes
    -----
    - Indices in `connect` are 0-based.
    - For the quadratic elements (tri6, quad8), this code automatically
      generates mid-edge nodes. The approach is uniform and assumes a
      structured rectangular grid. Each element cell places the extra
      mid-edge nodes by subdividing edges in half.
    """
    # Dispatch to the appropriate helper
    if ele_type == "D2_nn3_tri":
        return generate_tri3_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    elif ele_type == "D2_nn6_tri":
        return generate_tri6_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    elif ele_type == "D2_nn4_quad":
        return generate_quad4_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    elif ele_type == "D2_nn8_quad":
        return generate_quad8_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    else:
        raise ValueError(f"Unknown element type: {ele_type}")


# --------------------------------------------------------------------------
#   FUNCTIONS FOR EACH ELEMENT TYPE
# --------------------------------------------------------------------------

def generate_tri3_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a simple tri3 (3-node) mesh by subdividing each rectangular cell
    into two triangles.
    """
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    # Create the coordinates array
    coords_list = []
    for iy in range(n_nodes_y):
        for ix in range(n_nodes_x):
            xcoord = xl + ix * dx
            ycoord = yl + iy * dy
            coords_list.append((xcoord, ycoord))
    coords = np.array(coords_list, dtype=float)  # (n_nodes, 2)

    # Create the connectivity
    connectivity_list = []
    for iy in range(ny):
        for ix in range(nx):
            node0 = iy * n_nodes_x + ix
            node1 = iy * n_nodes_x + (ix + 1)
            node2 = (iy + 1) * n_nodes_x + ix
            node3 = (iy + 1) * n_nodes_x + (ix + 1)

            # two triangles
            connectivity_list.append([node0, node1, node2])
            connectivity_list.append([node2, node1, node3])

    connect = np.array(connectivity_list, dtype=int)
    return coords, connect


def generate_tri6_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a Tri6 (6-node) mesh by subdividing each rectangular cell into
    two triangles, with node ordering consistent with the standard Tri6
    shape functions N1..N6.
    """
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    # Refined grid has (2*nx + 1) points in x, (2*ny + 1) points in y
    npx = 2 * nx + 1
    npy = 2 * ny + 1

    # Build refined coordinates
    coords_list = []
    for iy in range(npy):
        for ix in range(npx):
            x = xl + 0.5 * ix * dx
            y = yl + 0.5 * iy * dy
            coords_list.append((x, y))
    coords = np.array(coords_list, dtype=float)

    def node_id(ix, iy):
        return iy * npx + ix

    connectivity_list = []

    for celly in range(ny):
        for cellx in range(nx):
            ix0 = 2 * cellx
            iy0 = 2 * celly

            # --- First triangle in the cell ---
            #
            #  Local reference:  N1(1,0), N2(0,1), N3(0,0)
            #
            #  In (x,y) space for the first half:
            #    * N1 = bottom-right corner
            #    * N2 = top-left corner
            #    * N3 = bottom-left corner
            #    * N4 = diagonal midpoint (N1->N2)
            #    * N5 = left-mid (N2->N3)
            #    * N6 = bottom-mid (N3->N1)
            #
            N1 = node_id(ix0+2, iy0  )  # bottom-right
            N2 = node_id(ix0,   iy0+2)  # top-left
            N3 = node_id(ix0,   iy0  )  # bottom-left
            N4 = node_id(ix0+1, iy0+1)  # midpoint diag (bottom-right -> top-left)
            N5 = node_id(ix0,   iy0+1)  # midpoint left vertical
            N6 = node_id(ix0+1, iy0  )  # midpoint bottom horizontal

            connectivity_list.append([N1, N2, N3, N4, N5, N6])

            # --- Second triangle in the cell ---
            #
            #  Local reference:  N1(1,0), N2(0,1), N3(0,0)
            #
            #  In (x,y) space for the second half:
            #    * N1 = top-right corner
            #    * N2 = top-left corner
            #    * N3 = bottom-right corner
            #    * N4 = top horizontal mid (N1->N2)
            #    * N5 = diagonal mid (N2->N3)
            #    * N6 = right vertical mid (N3->N1)
            #
            N1_2 = node_id(ix0+2, iy0+2) # top-right
            N2_2 = node_id(ix0,   iy0+2) # top-left
            N3_2 = node_id(ix0+2, iy0  ) # bottom-right
            N4_2 = node_id(ix0+1, iy0+2) # top horizontal midpoint
            N5_2 = node_id(ix0+1, iy0+1) # diagonal midpoint
            N6_2 = node_id(ix0+2, iy0+1) # right vertical midpoint

            connectivity_list.append([N1_2, N2_2, N3_2, N4_2, N5_2, N6_2])

    connect = np.array(connectivity_list, dtype=int)  # shape (n_elems, 6)
    return coords, connect


def generate_quad4_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a 2D mesh of 4-node quadrilaterals (bilinear quad).
    """
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    # Create node coordinates
    coords_list = []
    for iy in range(n_nodes_y):
        for ix in range(n_nodes_x):
            xcoord = xl + ix * dx
            ycoord = yl + iy * dy
            coords_list.append((xcoord, ycoord))
    coords = np.array(coords_list, dtype=float)  # (n_nodes, 2)

    # Connectivity
    connectivity_list = []
    for iy in range(ny):
        for ix in range(nx):
            node0 = iy * n_nodes_x + ix
            node1 = iy * n_nodes_x + (ix + 1)
            node2 = (iy + 1) * n_nodes_x + (ix + 1)
            node3 = (iy + 1) * n_nodes_x + ix
            # Quad element (node0, node1, node2, node3)
            connectivity_list.append([node0, node1, node2, node3])

    connect = np.array(connectivity_list, dtype=int)  # shape (n_elems, 4)
    return coords, connect


def generate_quad8_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a 2D mesh of 8-node quadrilaterals (quadratic quad).
    Each cell has corner + mid-edge nodes, excluding the central node.
    """
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx = 2 * nx + 1  # number of points in x-direction
    npy = 2 * ny + 1  # number of points in y-direction

    # Dictionary to map old node indices to new node indices
    node_map = {}
    new_coords = []
    new_index = 0

    # Build refined coordinates, skipping central nodes
    for iy in range(npy):
        for ix in range(npx):
            # Skip center nodes at (ix0+1, iy0+1) in 2x2 blocks
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            node_map[(ix, iy)] = new_index  # Store new index mapping
            new_coords.append((xl + 0.5 * ix * dx, yl + 0.5 * iy * dy))
            new_index += 1

    coords = np.array(new_coords, dtype=float)

    def node_id(ix, iy):
        return node_map[(ix, iy)]

    connectivity_list = []
    for celly in range(ny):
        for cellx in range(nx):
            ix0 = 2 * cellx
            iy0 = 2 * celly

            # Define the 8-node connectivity for the quadratic quadrilateral
            connectivity_list.append([
                node_id(ix0,   iy0),   # bottom-left
                node_id(ix0+2, iy0),   # bottom-right
                node_id(ix0+2, iy0+2), # top-right
                node_id(ix0,   iy0+2), # top-left
                node_id(ix0+1, iy0),   # mid-edge bottom
                node_id(ix0+2, iy0+1), # mid-edge right
                node_id(ix0+1, iy0+2), # mid-edge top
                node_id(ix0,   iy0+1)  # mid-edge left
            ])

    connect = np.array(connectivity_list, dtype=int)  # (n_elems, 8)
    return coords, connect


def mesh_outline(
    outline_points: list[tuple[float, float]],
    element_type: str,
    mesh_name: str,
    mesh_size: float = 0.05,
):
    """
    Generate a 2D mesh of the specified element type (D2_nn3_tri or D2_nn6_tri)
    for a user-defined shape outline using the gmsh Python API.

    Parameters
    ----------
    outline_points : list of (float, float)
        The polygon or spline points defining the shape's outline in XY.
        If not closed (first point != last point), the function appends
        the first point to the end.
    element_type : str
        Either 'D2_nn3_tri' (linear triangles) or 'D2_nn6_tri' (quadratic triangles).
    mesh_name : str
        A name for the gmsh model.
    mesh_size : float
        Characteristic length scale for the outline points.

    Returns
    -------
    coords : numpy.ndarray of shape (n_nodes, 2)
        The (x, y) coordinates of each node in the 2D mesh.
    connectivity : numpy.ndarray of shape (n_elems, n_nodes_per_elem)
        The triangular element connectivity (either 3 or 6 nodes/element),
        with 0-based node indices.

    Raises
    ------
    ValueError
        If an unsupported element_type is provided.
    RuntimeError
        If no elements of the requested type are found in the final mesh.
    """
    gmsh.initialize()
    gmsh.model.add(mesh_name)
    
    # Ensure the shape is properly closed
    if outline_points[0] != outline_points[-1]:
        outline_points.append(outline_points[0])
    
    # Create gmsh points
    point_tags = []
    for kk in range(0, len(outline_points) - 1):
        x = outline_points[kk][0]
        y = outline_points[kk][1]
        pt_tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size)
        point_tags.append(pt_tag)
    
    # Create lines
    curve_tags = []
    for i in range(len(point_tags) - 1):
        start_pt = point_tags[i]
        end_pt = point_tags[i + 1]
        line_tag = gmsh.model.geo.addLine(start_pt, end_pt)
        curve_tags.append(line_tag)
    
    start_pt = point_tags[-1]
    end_pt = point_tags[0]
    line_tag = gmsh.model.geo.addLine(start_pt, end_pt)
    curve_tags.append(line_tag)

    # Make a closed loop from these lines
    loop_tag = gmsh.model.geo.addCurveLoop(curve_tags)
    
    # Create a plane surface from the loop
    surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])
    
    # Optionally, define a physical group for the surface
    # so Gmsh understands it as a meshable 2D region.
    surf_group = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, surf_group, "MySurface")
    
    # Set element polynomial order
    if element_type == 'D2_nn3_tri':
        gmsh.model.mesh.setOrder(1)
        tri_wanted_type = 2   # Gmsh code for 3-node triangles
    elif element_type == 'D2_nn6_tri':
        gmsh.model.mesh.setOrder(2)
        tri_wanted_type = 9   # Gmsh code for 6-node (quadratic) triangles
    else:
        gmsh.finalize()
        raise ValueError(f"Unknown element type: {element_type}")
    
    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Generate 2D mesh
    gmsh.model.mesh.generate(dim=2)

    # Seems like a bug in gmsh, need to call this again if quadratic
    if element_type == 'D2_nn6_tri':
        gmsh.model.mesh.setOrder(2)

    # Ensure quadratic elements get generated
    gmsh.model.mesh.optimize()
    gmsh.model.mesh.renumberNodes()
    
    # Extract node coordinates and connectivity
    types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim=2, tag=surface_tag)
    
    # Find the index for the desired triangle type
    index_in_list = None
    for i, t in enumerate(types):
        if t == tri_wanted_type:
            index_in_list = i
            break
    if index_in_list is None:
        gmsh.finalize()
        raise RuntimeError(f"No elements of type {tri_wanted_type} found in mesh.")
    
    these_elem_tags = elem_tags[index_in_list]  # element IDs (not needed for connectivity)
    these_node_tags = node_tags[index_in_list]  # node IDs, flattened
    
    # Gmsh global nodes and their coordinates
    all_node_indices, all_node_coords, _ = gmsh.model.mesh.getNodes()
    # Build a map from gmsh node ID -> local index
    id2local = {node_id: i for i, node_id in enumerate(all_node_indices)}
    
    # Convert from (x,y,z) to (x,y)
    all_node_coords_3d = all_node_coords.reshape(-1, 3)
    coords = all_node_coords_3d[:, :2]
    
    # Build connectivity array
    n_nodes_per_elem = 3 if element_type == 'D2_nn3_tri' else 6
    n_elems = len(these_elem_tags)
    connectivity = np.zeros((n_elems, n_nodes_per_elem), dtype=int)
    
    # each element has n_nodes_per_elem node IDs in these_node_tags
    for e in range(n_elems):
        for k in range(n_nodes_per_elem):
            gmsh_node_id = these_node_tags[e * n_nodes_per_elem + k]
            connectivity[e, k] = id2local[gmsh_node_id]

    # Correct orientation element-by-element using signed area test
    for e in range(n_elems):
        elem_nodes = connectivity[e, :3]  # only first 3 are corners
        x1, y1 = coords[elem_nodes[0]]
        x2, y2 = coords[elem_nodes[1]]
        x3, y3 = coords[elem_nodes[2]]
        signed_area = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        if signed_area < 0:  # CW ordering, flip
            # flip corner node order
            connectivity[e, [1, 2]] = connectivity[e, [2, 1]]
            # flip mid-edge node order for quadratic elements
            if element_type == "D2_nn6_tri":
                connectivity[e, [3, 5]] = connectivity[e, [5, 3]]

    gmsh.finalize()
    return coords, connectivity


def get_terrier_outline():
    """
    Return a list of (x, y) coordinate pairs for a terrier head outline,
    as extracted from Inkscape.

    The coordinates below were copied directly from an Inkscape path export.
    You can further clean or scale them as needed.

    Returns
    -------
    outline_points : list of (float, float)
        The terrier outline, stored as a list of XY pairs.
    """
    # Raw coordinate string (from Inkscape)
    raw_coords = """
    88.016291,42.662919 -4.238281,-4.694941 -2.562257,-4.503359 -0.388221,-4.89158 
    2.562257,-4.11514 3.571632,-4.425715 4.037494,-4.037494 5.590376,-4.425715 
    7.143259,-4.8915802 5.51273,-3.8822063 2.40697,-1.242306 0.93173,0.4658647 
    0.0776,2.9504767 -2.25168,3.2610533 -4.81393,3.7269178 -5.59038,10.249025 
    -1.785813,5.124513 4.813933,-3.18341 7.68677,-2.950477 5.20216,0.854085 
    6.59975,2.950477 5.35744,3.493986 3.41635,0.465866 -1.16467,-2.251681 
    0.31058,-3.57163 3.10576,-3.882206 5.51274,-4.115139 8.46321,-4.037494 
    1.08702,0.310576 -1.16467,2.717545 -1.86346,2.251679 -0.93173,0.07764 
    -3.10576,6.755039 -2.71755,8.540854 6.36682,3.95985 6.6774,6.21153 
    4.03749,5.357445 1.55288,3.571629 -0.15529,3.416342 -1.39759,3.183408 
    -3.26105,4.348073 -0.15529,3.493984 -1.08702,3.804563 5.35745,3.804563 
    7.06561,2.795188 1.31995,2.872832 0.6988,6.755037 -0.38822,6.522109 
    -3.3387,8.152634 -2.32932,3.26106 -3.10577,2.40696 -1.70817,0.38822 
    -2.56225,2.17404 -2.71755,0.23293 -2.25168,2.25168 -2.01875,0.54351 
    -2.79518,-0.38822 -7.84206,3.3387 4.11514,4.11514 4.581,8.23027 
    2.25168,7.99735 0.6988,7.53148 -0.54351,3.57163 -2.40697,2.6399 
    -3.41634,0.23293 -8.38556,-2.6399 -7.06562,-2.79519 -5.59038,7.37619 
    -3.33869,3.10577 -5.74567,0.38822 -3.3387,-1.3976 -8.152629,-6.13388 
    -10.559603,-5.9786 -12.034837,-8.54085 -6.056244,-7.68677 -2.639898,-5.59038 
    -0.465866,-7.45383 0.232932,-6.05625 2.096391,-5.82331 2.872833,-2.795188 
    4.658648,-1.242306 1.009372,-1.941103 -7.220902,-7.842056 10.17138,1.708171 
    2.096391,-3.804563 4.270428,1.785816 4.658646,-9.006718 -2.562254,-4.89158 
    -4.348073,-5.668023 -1.708171,-6.133886 2.018747,-9.938445
    """
    # split the raw string by whitespace
    tokens = raw_coords.strip().split()
    # parse each token as "x,y"
    outline_points = []
    for t in tokens:
        x_str, y_str = t.split(",")
        x_val = float(x_str)
        y_val = float(y_str) * -1 # flipped for matplotlib plotting, image defaults is reveres
        if len(outline_points) == 0:
            outline_points.append((x_val, y_val))
        else:
            outline_points.append((x_val + outline_points[-1][0], y_val  + outline_points[-1][1]))
    outline_points.append(outline_points[0])
    return outline_points


def identify_rect_boundaries(
    coords: np.ndarray,
    connect: np.ndarray,
    ele_type: str,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    tol: float = 1e-10
):
    """
    Identify boundary nodes, elements, and faces for a rectangular 2D domain
    mesh. Boundaries are labeled as 'left', 'right', 'bottom', or 'top' based
    on coordinate checks against x_lower, x_upper, y_lower, y_upper.

    Parameters
    ----------
    coords : numpy.ndarray of shape (n_nodes, 2)
        The node coordinates array, typically from generate_rect_mesh_2d(...).
    connect : numpy.ndarray
        The element connectivity array, shape depends on ele_type:
          - tri3  -> (n_elems, 3)
          - tri6  -> (n_elems, 6)
          - quad4 -> (n_elems, 4)
          - quad8 -> (n_elems, 8)
    ele_type : str
        One of {'D2_nn3_tri', 'D2_nn6_tri', 'D2_nn4_quad', 'D2_nn8_quad'}.
    x_lower, x_upper : float
        The domain boundaries in x.
    y_lower, y_upper : float
        The domain boundaries in y.
    tol : float, optional
        Tolerance for comparing floating-point coordinates. If a node is
        within `tol` of a boundary, it's considered on that boundary.

    Returns
    -------
    boundary_nodes : dict of {str -> set of int}
        Keys are 'left','right','bottom','top'. Values are sets of node indices
        that lie on that boundary.
    boundary_edges : dict of {str -> list of (elem_id, face_id)}
        Keys are 'left','right','bottom','top'. Each entry is a list of
        tuples (element_id, local_face_id) indicating which element-face
        belongs to that boundary.

    Notes
    -----
    - For triangular elements, each face/edge is defined by consecutive nodes
      in the connectivity. For tri3, edges are (0,1), (1,2), (2,0); for tri6,
      edges are (0,1,3), (1,2,4), (2,0,5).
    - For quadrilateral elements, each face is defined by consecutive nodes
      in the connectivity array. For quad4, faces are (0,1), (1,2), (2,3), (3,0);
      for quad8, faces are (0,1,4), (1,2,5), (2,3,6), (3,0,7).
    - This function focuses on a strictly rectangular domain. We identify
      boundary nodes by checking x or y vs. x_lower, x_upper, y_lower, y_upper
      within a tolerance. Then, we find which element edges/faces connect
      these boundary nodes to label them accordingly.
    """

    n_nodes = coords.shape[0]
    n_elems = connect.shape[0]

    # 1. Identify boundary nodes by coordinate
    #    We'll store them in sets to avoid duplicates
    left_nodes = set()
    right_nodes = set()
    bottom_nodes = set()
    top_nodes = set()

    for nid in range(n_nodes):
        xval, yval = coords[nid]
        # Compare with tolerance
        if abs(xval - x_lower) < tol:
            left_nodes.add(nid)
        if abs(xval - x_upper) < tol:
            right_nodes.add(nid)
        if abs(yval - y_lower) < tol:
            bottom_nodes.add(nid)
        if abs(yval - y_upper) < tol:
            top_nodes.add(nid)

    # 2. Determine how faces are enumerated for each element type
    #    We'll define a helper that, given 'ele_type', returns a list of "faces"
    #    as tuples of local node indices in the connectivity array.
    face_definitions = local_faces_for_element_type(ele_type)

    # 3. Identify boundary edges/faces by checking if *all* the nodes
    #    in that face belong to the same boundary set. Because if an entire
    #    face is on x_lower => all face nodes must have x ~ x_lower, etc.
    #    We'll store the result as a dict: { boundary : list of (elem_id, face_id) }
    boundary_edges = {
        'left': [],
        'right': [],
        'bottom': [],
        'top': []
    }

    for e in range(n_elems):
        # Each face is a list of local node indices in the connectivity
        for face_id, face_lnodes in enumerate(face_definitions):
            # The actual global node ids for this face
            face_nodes = connect[e, face_lnodes]
            # We'll see if they are on left, right, bottom, top.
            # In a rectangular domain, if all these face nodes are in left_nodes,
            # that face is a left boundary, etc. But watch out for corner elements
            # that might belong to left *and* bottom, for instance.
            # Typically, we consider it "left" if all face nodes are in left_nodes
            # We'll do it that way for simplicity. 
            # A corner face might appear in two boundary sets if it's degenerate, 
            # but usually an element face won't be "on" two boundaries at once 
            # unless the domain is extremely coarse.
            if all(fn in left_nodes for fn in face_nodes):
                boundary_edges['left'].append((e, face_id))
            if all(fn in right_nodes for fn in face_nodes):
                boundary_edges['right'].append((e, face_id))
            if all(fn in bottom_nodes for fn in face_nodes):
                boundary_edges['bottom'].append((e, face_id))
            if all(fn in top_nodes for fn in face_nodes):
                boundary_edges['top'].append((e, face_id))

    # 4. Return the results
    boundary_nodes = {
        'left': left_nodes,
        'right': right_nodes,
        'bottom': bottom_nodes,
        'top': top_nodes
    }

    return boundary_nodes, boundary_edges


def local_faces_for_element_type(ele_type: str):
    """
    Return a list of "faces" for the given 2D element type, where each
    face is defined by a tuple of local connectivity indices.
    
    For example, tri3 has 3 edges:
       face0 = (0,1)
       face1 = (1,2)
       face2 = (2,0)

    tri6 (quadratic triangle) has 3 edges each with 3 nodes:
       face0 = (0,1,3)
       face1 = (1,2,4)
       face2 = (2,0,5)

    quad4 has 4 edges:
       face0 = (0,1)
       face1 = (1,2)
       face2 = (2,3)
       face3 = (3,0)

    quad8 (quadratic quad) has 4 edges each with 3 nodes:
       face0 = (0,1,4)
       face1 = (1,2,5)
       face2 = (2,3,6)
       face3 = (3,0,7)
    """
    if ele_type == "D2_nn3_tri":
        # 3-node triangle
        return [
            (0, 1),
            (1, 2),
            (2, 0)
        ]
    elif ele_type == "D2_nn6_tri":
        # 6-node triangle
        return [
            (0, 1, 3),
            (1, 2, 4),
            (2, 0, 5)
        ]
    elif ele_type == "D2_nn4_quad":
        # 4-node quad
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0)
        ]
    elif ele_type == "D2_nn8_quad":
        # 8-node quad
        return [
            (0, 1, 4),
            (1, 2, 5),
            (2, 3, 6),
            (3, 0, 7)
        ]
    else:
        raise ValueError(f"Unknown element type: {ele_type}")


def assign_fixed_nodes_rect(
    boundary_nodes: dict[str, set[int]],
    boundary: str,
    dof_0_disp: float = None,
    dof_1_disp: float = None,
    dof_2_disp: float = None
) -> np.ndarray:
    """
    Build a (3, n_fixed) array of prescribed boundary conditions for all nodes
    on a specified boundary of a rectangular 2D mesh.

    Parameters
    ----------
    boundary_nodes : dict of {str -> set of int}
        A dictionary mapping each boundary ('left','right','bottom','top') to 
        a set of node indices on that boundary.
    boundary : str
        Which boundary name in boundary_nodes to apply these DOF constraints to 
        (e.g. 'left', 'top', etc.).
    dof_0_disp : float or None, optional
        If not None, fix DOF #0 of each node at the given displacement.
    dof_1_disp : float or None, optional
        If not None, fix DOF #1 of each node at the given displacement.
    dof_2_disp : float or None, optional
        If not None, fix DOF #2 of each node at the given displacement.
        In a 2D problem, typically dof_2_disp is None by default.

    Returns
    -------
    fixed_nodes : numpy.ndarray, shape (3, n_fixed)
        The prescribed boundary conditions. Each column has:
          [ node_id, dof_index, displacement_value ].

    Notes
    -----
    - Only DOFs for which a non-None displacement is provided will be fixed.
    - For 2D (ncoord=2, ndof=2), typically dof_2_disp is unused.
    - If boundary_nodes[boundary] is empty, this function returns an empty array.
    """
    # Get all node indices on the specified boundary
    node_ids = boundary_nodes.get(boundary, set())
    if not node_ids:
        # No nodes on this boundary => return empty array
        return np.empty((3, 0), dtype=float)

    # Build a list of constraints
    constraints = []
    for node_id in node_ids:
        if dof_0_disp is not None:
            constraints.append((node_id, 0, dof_0_disp))
        if dof_1_disp is not None:
            constraints.append((node_id, 1, dof_1_disp))
        if dof_2_disp is not None:
            constraints.append((node_id, 2, dof_2_disp))

    # If no constraints were added (all disp = None), return empty
    if not constraints:
        return np.empty((3, 0), dtype=float)

    # Convert list to numpy array and transpose to shape (3, n_fixed)
    fixed_array = np.array(constraints, dtype=float).T  # shape => (3, n_fixed)
    return fixed_array


def assign_uniform_load_rect(
    boundary_edges: dict[str, list[tuple[int, int]]],
    boundary: str,
    dof_0_load: float = 0.0,
    dof_1_load: float = 0.0,
    dof_2_load: float = 0.0
) -> np.ndarray:
    """
    Create a distributed-load specification for a boundary in a 2D or 3D mesh,
    returning an array dload_info of shape (ndof+2, n_face_loads).

    Each column of dload_info describes a uniform traction load on a single
    element-face along the specified boundary. The format:
      - dload_info[0, j] => element index (elem_id)
      - dload_info[1, j] => local face ID (face_id) on that element
      - dload_info[2, j], dload_info[3, j], [dload_info[4, j]] => the traction
        components for dof=0,1,[2].

    Parameters
    ----------
    boundary_edges : dict of {str -> list of (elem_id, face_id)}
        Keys are 'left','right','bottom','top'. Each entry is a list of
        (element_id, local_face_id) pairs indicating which element-face
        belongs to that boundary.
    boundary : str
        The boundary name in boundary_edges to which the uniform traction
        is applied (e.g. 'left', 'top', etc.).
    dof_0_load : float, optional
        The traction in the dof=0 direction (e.g., x-direction in 2D).
    dof_1_load : float, optional
        The traction in the dof=1 direction (e.g., y-direction in 2D).
    dof_2_load : float, optional
        The traction in the dof=2 direction (if 3D). If you are strictly 2D,
        this should be 0 (the default).

    Returns
    -------
    dload_info : numpy.ndarray, shape (ndof+2, n_face_loads)
        The distributed face load info. Each column corresponds to a single face
        along `boundary`. The top rows contain the (element_id, face_id),
        followed by the traction components. If no boundary faces exist or the
        traction is zero in all directions and you prefer to omit them, you can
        filter accordingly.

    Notes
    -----
    - If dof_2_load is nonzero, we assume ndof=3. Otherwise, ndof=2.
    - If the boundary has no faces in boundary_edges[boundary], returns an
      empty array with shape (ndof+2, 0).
    - In a typical 2D code with tri or quad elements, face_id might range
      from 0..2 or 0..3, etc.
    - The traction is uniform. If you want a variable traction, you might
      compute different values per face.
    """
    # check boundary faces
    faces = boundary_edges.get(boundary, [])
    n_face_loads = len(faces)
    ndof = 3  # assumes ndof = 3, if code is 2D final entry will be ignored
    if n_face_loads == 0:
        # no faces => return empty
        return np.empty((ndof+2, 0), dtype=float)

    # build the traction vector for each face
    # shape => (ndof+2, n_face_loads)
    dload_info = np.zeros((ndof+2, n_face_loads), dtype=float)

    # dof loads in a list
    load_list = [dof_0_load, dof_1_load, dof_2_load]

    # iterate through faces and add to dload_info vector
    for j, (elem_id, face_id) in enumerate(faces):
        # element_id, face_id
        dload_info[0, j] = elem_id
        dload_info[1, j] = face_id
        # traction components
        for i in range(ndof):
            dload_info[i + 2, j] = load_list[i]

    return dload_info
