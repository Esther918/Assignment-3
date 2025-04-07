from finiteelementanalysis import discretization as di
from finiteelementanalysis import local_element as loc_el
import numpy as np
from scipy.sparse import coo_matrix


def global_stiffness(
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: list,
    displacement: np.ndarray
):
    """
    Assemble the global stiffness matrix for a finite element model.

    The function infers the dimensionality (ncoord), the number of degrees of
    freedom per node (ndof), and the number of local element nodes (nelnodes)
    from the element type, then loops over all elements to form and accumulate
    each element's stiffness contribution into a global stiffness matrix.

    Parameters
    ----------
    ele_type : str
        The element type identifier (e.g., 'D3_nn8_hex'). Used to look up
        dimension, DOFs, local node count, etc.
    coords : numpy.ndarray of shape (ncoord, nnode)
        Global coordinates of all nodes. coords[i, a] is the i-th coordinate of
        node a (0-based). The number of columns, nnode, is the total number of
        nodes in the mesh.
    connect : numpy.ndarray of shape (nelnodes, nelem)
        Element connectivity array, where connect[a, e] is the global node index
        of the a-th local node of the e-th element. Assumed to be 0-based 
        indexing.
    material_props : list
        Material property array, e.g., for a Neo-Hookean model [mu, bulk_modulus].
    displacement : numpy.ndarray of shape (ndof, nnode)
        The global displacement vector, matching the format of coords.

    Returns
    -------
    K_global : numpy.ndarray of shape (ndof*nnode, ndof*nnode)
        The assembled global stiffness matrix.

    Notes
    -----
    - Each element's stiffness matrix is computed by a local function
      (here called `element_stiffness`), then added into `K_global`.
    - The mesh size (nnode, nelem) is inferred from `coords.shape[1]` and
      `connect.shape[1]`, respectively.
    - The local node coordinates and local displacement are extracted from
      the global arrays and passed to `element_stiffness` for each element.
    """

    ncoord, ndof, nelnodes = di.element_info(ele_type)
    nnode = coords.shape[1]
    nelem = connect.shape[1]
    K_global = np.zeros((ndof * nnode, ndof * nnode))

    for e in range(nelem):
        node_indices = connect[:, e]
        ele_coords = coords[:, node_indices]         # shape: (ncoord, nelnodes)
        ele_disp = displacement[:, node_indices]     # shape: (ndof, nelnodes)

        k_element = loc_el.element_stiffness(ele_type, material_props, ele_coords, ele_disp)

        # Compute global DOF indices for the element
        dof_indices = np.ravel([ndof * node + np.arange(ndof) for node in node_indices])

        for a in range(nelnodes * ndof):
            for b in range(nelnodes * ndof):
                K_global[dof_indices[a], dof_indices[b]] += k_element[a, b]

    return K_global


##############################################################################
# OLD VERSION OF THE FUNCTION THAT IS NOT VECTORIZED
##############################################################################
# def global_stiffness(
#     ele_type: str,
#     coords: np.ndarray,
#     connect: np.ndarray,
#     material_props: list,
#     displacement: np.ndarray
# ):
#     # Retrieve key element info: ncoord, ndof, nelnodes
#     ncoord, ndof, nelnodes = di.element_info(ele_type)

#     # Number of global nodes and elements
#     nnode = coords.shape[1]
#     nelem = connect.shape[1]

#     # Allocate the global stiffness matrix
#     K_global = np.zeros((ndof * nnode, ndof * nnode))

#     # Loop over elements
#     for e in range(nelem):
#         # Temporary arrays for local coordinates (shape=(ncoord, nelnodes))
#         # and local displacement (shape=(ndof*nelnodes, 1))
#         ele_coords = np.zeros((ncoord, nelnodes))
#         ele_disp = np.zeros((ndof, nelnodes))

#         # Gather local node coordinates and local displacements
#         for a in range(nelnodes):
#             global_node = connect[a, e]  # 0-based index
#             for i in range(ncoord):
#                 ele_coords[i, a] = coords[i, global_node]
#             for i in range(ndof):
#                 ele_disp[i, a] = displacement[i, global_node]

#         # Compute the element stiffness matrix
#         k_element = loc_el.element_stiffness(ele_type, material_props, ele_coords, ele_disp)

#         # Assemble into the global matrix
#         for a in range(nelnodes):
#             global_node_a = connect[a, e]
#             for i in range(ndof):
#                 row = ndof * global_node_a + i
#                 for b in range(nelnodes):
#                     global_node_b = connect[b, e]
#                     for k in range(ndof):
#                         col = ndof * global_node_b + k
#                         K_global[row, col] += k_element[ndof * a + i, ndof * b + k]

#     return K_global


def global_stiffness_sparse(
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: list,
    displacement: np.ndarray
):
    """
    Assemble the global stiffness matrix for a finite element model.

    The function infers the dimensionality (ncoord), the number of degrees of
    freedom per node (ndof), and the number of local element nodes (nelnodes)
    from the element type, then loops over all elements to form and accumulate
    each element's stiffness contribution into a global stiffness matrix.

    Parameters
    ----------
    ele_type : str
        The element type identifier (e.g., 'D3_nn8_hex'). Used to look up
        dimension, DOFs, local node count, etc.
    coords : numpy.ndarray of shape (ncoord, nnode)
        Global coordinates of all nodes. coords[i, a] is the i-th coordinate of
        node a (0-based). The number of columns, nnode, is the total number of
        nodes in the mesh.
    connect : numpy.ndarray of shape (nelnodes, nelem)
        Element connectivity array, where connect[a, e] is the global node index
        of the a-th local node of the e-th element. Assumed to be 0-based 
        indexing.
    material_props : list
        Material property array, e.g., for a Neo-Hookean model [mu, bulk_modulus].
    displacement : numpy.ndarray of shape (ndof, nnode)
        The global displacement vector, matching the format of coords.

    Returns
    -------
    K_global : numpy.ndarray of shape (ndof*nnode, ndof*nnode)
        The assembled global stiffness matrix.

    Notes
    -----
    - Each element's stiffness matrix is computed by a local function
      (here called `element_stiffness`), then added into `K_global`.
    - The mesh size (nnode, nelem) is inferred from `coords.shape[1]` and
      `connect.shape[1]`, respectively.
    - The local node coordinates and local displacement are extracted from
      the global arrays and passed to `element_stiffness` for each element.
    """
    ncoord, ndof, nelnodes = di.element_info(ele_type)
    nnode = coords.shape[1]
    nelem = connect.shape[1]

    # Preallocate lists for COO format (i, j, val)
    row_inds = []
    col_inds = []
    data_vals = []

    for e in range(nelem):
        node_indices = connect[:, e]  # shape: (nelnodes,)
        ele_coords = coords[:, node_indices]         # (ncoord, nelnodes)
        ele_disp = displacement[:, node_indices]     # (ndof, nelnodes)

        k_element = loc_el.element_stiffness(ele_type, material_props, ele_coords, ele_disp)

        # Flattened global DOF indices for this element
        dof_indices = np.ravel([ndof * node + np.arange(ndof) for node in node_indices])

        # Assemble element stiffness into global sparse matrix data
        for a in range(nelnodes * ndof):
            for b in range(nelnodes * ndof):
                row_inds.append(dof_indices[a])
                col_inds.append(dof_indices[b])
                data_vals.append(k_element[a, b])

    # Assemble global sparse stiffness matrix
    K_global_sparse = coo_matrix((data_vals, (row_inds, col_inds)),
                                 shape=(ndof * nnode, ndof * nnode))

    return K_global_sparse


def global_traction(
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    dload_info: np.ndarray,
) -> np.ndarray:
    """
    Assemble the global traction (load) vector for specified element faces 
    carrying a uniform traction load in a finite element model.

    This function loops over a set of prescribed face loads, extracts the 
    local face coordinates and DOF mapping from global arrays, and calls 
    a local routine (e.g., `loc_el.element_distributed_load`) to compute 
    the per-face load. It then assembles each face load contribution into 
    a global vector.

    Parameters
    ----------
    ele_type : str
        Element type identifier (e.g., 'D3_nn8_hex'), used to retrieve the
        spatial dimension (ncoord), DOFs per node (ndof), and face information
        from a data interface `di.element_info` and `di.face_info`.
    coords : numpy.ndarray of shape (ncoord, n_nodes)
        Global coordinates of all nodes in the mesh. `coords[i, a]` is the i-th
        coordinate of node a.
    connect : numpy.ndarray of shape (nodes_per_ele, n_elem)
        The connectivity array: `connect[a, e]` is the global node index of
        the a-th local node of element e.
    dload_info : numpy.ndarray, shape (ndof+2, n_face_loads)
        An array describing distributed face loads, where each column corresponds
        to one face load specification:
          - dload_info[0, j]: The element index (0-based) containing the face.
          - dload_info[1, j]: The face identifier (e.g., 0..5 for a hex).
          - dload_info[i+2, j] for i in [0..(ndof-1)]: The traction components
            on that face (e.g., tx, ty, [tz] if 3D).
        The traction is assumed uniform across the face.

    Returns
    -------
    F_global : numpy.ndarray of shape (ndof * n_nodes,)
        The assembled global traction (load) vector. Each DOF entry accumulates
        the contributions from any faces that carry a prescribed traction.

    Notes
    -----
    - The function calls `di.element_info(ele_type)` to get (ncoord, ndof, _).
    - It calls `di.face_info(ele_type, face)` to get the number of face nodes and
      the local node indices that define that face (`nodes_on_face`).
    - It then calls a local element routine 
      (`loc_el.element_distributed_load(ele_type, coords, traction_vec)`) to 
      compute the per-face load vector.
    - dofs : For a node `a` and DOF `i`, the
        global index is `ndof * a + i`. This is used to map from local dofs to
        global positions in the traction vector.
    - Finally, it assembles that face load vector into `global_traction_vector`.
    - Be sure `connect` and `dofs` are in 0-based indexing.
    """
    # get element information
    ncoord, ndof, _ = di.element_info(ele_type)

    # Allocate the global traction vector
    nnodes = coords.shape[1]
    F_global = np.zeros(ndof * nnodes)


    # Loop over each prescribed load
    ndload = dload_info.shape[1]
    for load_idx in range(ndload):
        # Temporary array for traction
        traction_vec = np.zeros(ndof)
        # Extract element index and face ID from dload_info
        element_idx = int(dload_info[0, load_idx])
        face = int(dload_info[1, load_idx])

        # Number of face nodes and local node indices on that face
        # (di.face_info might return something like: (face_elem_type, num_face_nodes, nodes_on_face))
        _, num_face_nodes, nodes_on_face = di.face_info(ele_type, face)

        # Prepare local arrays for face nodes
        element_coord = np.zeros((ncoord, num_face_nodes), dtype=float)
        element_dof = np.zeros((ndof, num_face_nodes), dtype=int)

        # Extract coords and dofs for the nodes on this element face
        for a in range(num_face_nodes):
            global_node = connect[nodes_on_face[a], element_idx]
            for i in range(ncoord):
                element_coord[i, a] = coords[i, global_node]
            for i in range(ndof):
                element_dof[i, a] = ndof * global_node + i

        # Build the traction vector from the row(s) of dload_info
        for i in range(ndof):
            traction_vec[i] = dload_info[i + 2, load_idx]

        # Compute the element face load vector
        rel = loc_el.element_distributed_load(ele_type, element_coord, traction_vec)

        # Assemble into the global load vector
        for a in range(num_face_nodes):
            global_node = connect[nodes_on_face[a], element_idx]
            for i in range(ndof):
                rw = ndof * global_node + i
                F_global[rw] += rel[ndof * a + i]

    return F_global


def global_residual(
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: list,
    displacement: np.ndarray
) -> np.ndarray:
    """
    Assemble the global residual (internal force) vector for all elements
    in a finite element model, given the global displacement and mesh data.

    This function determines the number of coordinates (ncoord), DOFs per node 
    (ndof), and element node count (nelnodes) from `ele_type`, then loops 
    over each element. For each element, it extracts the local nodal coordinates 
    and DOF indices, calls a local routine (`loc_el.element_residual`) to compute 
    the per-element residual, and finally accumulates these contributions into 
    a global residual vector.

    Parameters
    ----------
    ele_type : str
        The element type identifier (e.g., 'D3_nn8_hex'). Used to look up:
        - The spatial dimension (ncoord),
        - The number of DOFs per node (ndof),
        - The number of element nodes (nelnodes),
        via a data interface (e.g. `di.element_info(ele_type)`).
    coords : numpy.ndarray of shape (ncoord, n_nodes)
        Global coordinates of all nodes. 
        `coords[i, a]` is the i-th coordinate of node a (0-based indexing).
    connect : numpy.ndarray of shape (max_nodes_per_elem, n_elems)
        Element connectivity array. `connect[a, e]` is the global node index 
        for the a-th local node of the e-th element (0-based indexing).
    material_props : list
        A list or array of material properties (e.g., [mu, bulk_modulus]) 
        required by the local element routine.
    displacement : numpy.ndarray of shape (ndof, nnode)
        displacement : numpy.ndarray of shape (ndof, nnode)
        The global displacement vector, matching the format of coords.

    Returns
    -------
    R_global : np.ndarray of shape (ndof * n_nodes)
        The assembled global residual (internal force) vector, where each 
        element's contribution is summed according to the mesh connectivity.

    Notes
    -----
    - The function obtains `ncoord`, `ndof`, and `nelnodes` from 
      `di.element_info(ele_type)`.
    - It then loops over each element (the total number of elements is 
      `connect.shape[1]`), gathering local nodal information and computing 
      an element residual via `loc_el.element_residual(...)`.
    - The local residual is assumed to have size `(ndof * n)`, where `n` is 
      the number of nodes for that element.
    - Indices are all 0-based.
    """
    # get mesh and element information
    nnode = coords.shape[1]          # number of global nodes
    nelem = connect.shape[1]         # number of elements
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # Initialize the global residual vector
    R_global = np.zeros(ndof * nnode)
 
    # Loop over all elements
    for e in range(nelem):
        # Temporary arrays for local coordinates and local displacements
        element_coord = np.zeros((ncoord, nelnodes))
        element_disp = np.zeros((ndof, nelnodes))
        # Gather local node coordinates and local displacements
        for a in range(nelnodes):
            global_node = connect[a, e]
            # Coordinates
            for i in range(ncoord):
                element_coord[i, a] = coords[i, global_node]
            # Displacements
            for i in range(ncoord):
                element_disp[i, a] = displacement[i, global_node]

        # Compute the local residual for this element
        R_element = loc_el.element_residual(ele_type, element_coord, material_props, element_disp)

        # Accumulate into the global residual
        for a in range(nelnodes):
            global_node = connect[a, e]
            for i in range(ndof):
                row = ndof * global_node + i
                R_global[row] += R_element[ndof * a + i]

    return R_global
