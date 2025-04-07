from finiteelementanalysis import discretization as di
import numpy as np
from numba import njit


def element_residual(ele_type, coords, materialprops, displacement):
    """
    Assemble the element residual (internal force) vector for a hyperelastic 
    large-deformation element, in a more modular style using helper functions.

    Parameters
    ----------
    ele_type : str
        The element type identifier (e.g., 'D3_nn8_hex'), which determines 
        the spatial dimension (ncoord), DOFs (ndof), number of element nodes,
        integration scheme, and shape function details.
    coords : numpy.ndarray of shape (ncoord, nelnodes)
        The global coordinates of the element's nodes. For example, in 2D, 
        coords[0,:] and coords[1,:] hold the x- and y-coordinates, respectively.
    materialprops : array_like
        Material property array (e.g., [mu, k] for a Neo-Hookean material).
    displacement : numpy.ndarray of shape (ndof, nelnodes)
        The displacement at each node of the element. For instance, in 2D 
        displacement[0,a] is the x-displacement of node a, displacement[1,a] is the y-displacement.

    Returns
    -------
    rel : numpy.ndarray of shape (ndof * nelnodes,)
        The assembled element residual (internal force) vector.

    Notes
    -----
    - This version calls a set of small helper functions to compute the Jacobian,
      the deformation gradient, the left Cauchy-Green tensor, etc.
    - Integration points and shape function details are assumed to be provided by
      a data interface (`di`), which includes shape_fcn_derivative, integration_info, etc.
    """
    # Get dimensional and node info for this element type
    ncoord, ndof, nelnodes = di.element_info(ele_type)

    # Retrieve integration data (number of points, local coords, weights)
    num_integration_points, xi_array, w_array = di.integration_info(ele_type)

    # Allocate the element residual vector
    rel = np.zeros(ndof * nelnodes)

    for intpt in range(num_integration_points):
        # Local coordinates at this integration point
        xi = xi_array[:, intpt]  # shape (ncoord,)

        # Compute shape function derivatives in local coords, dNdxi: (nelnodes, ncoord)
        dNdxi = di.shape_fcn_derivative(ele_type, xi)

        # 1) Compute the Jacobian matrix (dxdxi) and determinant (dt).
        #    'coords' is (ncoord, nelnodes), 'dNdxi' is (nelnodes, ncoord).
        #    Thus coords @ dNdxi => (ncoord, ncoord).
        dxdxi, det_dxdxi = compute_jacobian(coords, dNdxi)

        # 2) Compute the inverse of dxdxi, used to convert derivatives to global coords
        dxidx = np.linalg.inv(dxdxi)

        # 3) Convert shape function derivatives to global coords
        #    dNdx: (nelnodes, ncoord) = dNdxi @ dxidx
        dNdx = convert_derivatives(dNdxi, dxidx)

        # 4) Compute the deformation gradient, F = I + displacement @ dNdx
        #    Here 'displacement' is (ncoord, nelnodes), 'dNdx' is (nelnodes, ncoord).
        F = compute_deformation_gradient(ncoord, displacement, dNdx)

        # 5) Compute J = det(F) and B = F * F^T
        J = compute_J(F)
        B = compute_B(F)

        # 6) Convert shape function derivatives to spatial coords:
        #    dNdxs = dNdx @ Finv
        Finv = compute_Finv(F)
        dNdxs = convert_to_spatial_derivatives(dNdx, Finv)

        # 7) Compute the Kirchhoff stress
        stress = kirchhoff_stress(B, J, materialprops)

        # 8) Accumulate the element residual
        for a in range(nelnodes):
            for i in range(ndof):
                row = ndof * a + i
                for j in range(ncoord):
                    rel[row] += stress[i, j] * dNdxs[a, j] * w_array[intpt][0] * det_dxdxi

    return rel


def element_distributed_load(ele_type, coords, traction):
    """
    Compute the distributed (surface) load vector for an element face subjected
    to a prescribed traction in 2D or 3D, using a structure similar to 
    'compute_stiffness_contributions'.

    Parameters
    ----------
    ele_type : str
        The element type identifier for the parent element (e.g., 'D3_nn8_hex').
        Used to look up dimension (ncoord), DOFs (ndof), face element type, etc.
    coords : numpy.ndarray, shape (ncoord, num_face_nodes)
        Coordinates of the face's nodes.
    traction : array_like of length ndof
        The applied traction vector on this face, e.g., [tx, ty, tz] in 3D.

    Returns
    -------
    r : numpy.ndarray of shape (ndof * num_face_nodes,)
        The assembled load vector corresponding to the distributed traction
        on this face.

    Notes
    -----
    - The "det_dxdxi" here represents the line length (2D) or area (3D), 
      matching the conceptual usage in stiffness assembly code.
    - The per-integration-point load contribution is computed by 
      'compute_face_load_contribution', then scaled by weight * det_dxdxi
      in a single line, as in 'compute_stiffness_contributions'.
    """

    # 1. Retrieve element dimension & DOF info
    ncoord, ndof, _ = di.element_info(ele_type)
    face_elem_type, num_face_nodes, _ = di.face_info(ele_type)

    # 2. Retrieve integration data: (#points, local coords xi_array, weights w_array)
    num_integration_points, xi_array, w_array = di.integration_info(face_elem_type)

    # 3. Allocate the load vector
    r = np.zeros(ndof * num_face_nodes)

    # 4. Integration loop
    for intpt in range(num_integration_points):
        # (a) local face coordinate
        xi = xi_array[:, intpt]

        # (b) shape functions & derivatives
        N = di.shape_fcn(face_elem_type, xi)                
        dNdxi = di.shape_fcn_derivative(face_elem_type, xi)  

        # (c) Jacobian for a face
        dxdxi = compute_face_jacobian(ncoord, coords, dNdxi)

        # (d) Face measure (length in 2D, area in 3D) = "det_dxdxi"
        det_dxdxi = compute_face_measure(ncoord, dxdxi)

        # (e) Per-point load contribution
        r_local = compute_face_load_contribution(num_face_nodes, ndof, traction, N)

        # (f) Accumulate in the style: r += w * det_dxdxi * contribution
        r += w_array[intpt] * det_dxdxi * r_local

    return r


def compute_face_jacobian(ncoord, coords, dNdxi):
    """
    Compute the Jacobian matrix (dxdxi) for a face (line in 2D or surface in 3D).

    Parameters
    ----------
    ncoord : int
        Spatial dimension (2 for 2D, 3 for 3D).
    coords : numpy.ndarray, shape (ncoord, num_face_nodes)
        Coordinates of the face nodes.
    dNdxi : numpy.ndarray, shape (num_face_nodes, ncoord - 1)
        Shape function derivatives wrt local face coordinates.

    Returns
    -------
    dxdxi : numpy.ndarray, shape (ncoord, ncoord - 1)
        The Jacobian matrix relating local face coordinates (xi) to global coordinates.
    """
    num_face_nodes = dNdxi.shape[0]
    dxdxi = np.zeros((ncoord, ncoord - 1))
    for i in range(ncoord):
        for j in range(ncoord - 1):
            for a in range(num_face_nodes):
                dxdxi[i, j] += coords[i, a] * dNdxi[a, j]
    return dxdxi


def compute_face_measure(ncoord, dxdxi):
    """
    Compute the measure (line length in 2D, surface area in 3D) from the face Jacobian.

    Parameters
    ----------
    ncoord : int
        Spatial dimension (2).
    dxdxi : numpy.ndarray, shape (ncoord, ncoord - 1)
        Jacobian matrix (non-square in most cases).

    Returns
    -------
    det_dxdxi : float
        The measure of the face element at the integration point 
        (length in 2D, area in 3D).

    Raises
    ------
    ValueError
        If ncoord is not 2 or 3.
    """
    if ncoord == 2:
        col = dxdxi[:, 0]
        det_dxdxi = np.sqrt(col[0]**2 + col[1]**2)  # line length
    else:
        raise ValueError(f"Unsupported dimension ncoord={ncoord}.")
    return det_dxdxi


def compute_face_load_contribution(num_face_nodes, ndof, traction, N):
    """
    Compute the contribution to the load vector from a single integration point.

    Parameters
    ----------
    num_face_nodes : int
        Number of nodes on the face.
    ndof : int
        Number of degrees of freedom per node.
    traction : array_like of length ndof
        The traction vector (e.g., [tx, ty] in 2D or [tx, ty, tz] in 3D).
    N : numpy.ndarray of shape (num_face_nodes,)
        Shape function values at this integration point.

    Returns
    -------
    r_local : numpy.ndarray of shape (ndof * num_face_nodes,)
        The local (per-integration-point) contribution to the load vector.
    """
    r_local = np.zeros(ndof * num_face_nodes)
    for a in range(num_face_nodes):
        for i in range(ndof):
            row = ndof * a + i
            r_local[row] = N[a][0] * traction[i]
    return r_local


def element_stiffness(ele_type: str, material_props, coords: np.ndarray, displacement: np.ndarray):
    """
    Computes the element stiffness matrix.

    Args:
        ncoord (int): Number of coordinates (2D or 3D).
        ndof (int): Number of degrees of freedom per node.
        nelnodes (int): Number of nodes per element.
        elident (int): Element identifier (not used here).
        coords (np.ndarray): (ncoord, nelnodes) array of node coordinates.
        material_props numpy array containing material properties (e.g., mu, kappa).
        displacement (np.ndarray): (ndof, nelnodes) array of nodal displacements.

    Returns:
        np.ndarray: Element stiffness matrix (ndof * nelnodes, ndof * nelnodes).
    """
    #  Get dicretization specific information
    (ncoord, ndof, nelnodes) = di.element_info(ele_type)
    (num_integration_points, xi_array, w_array) = di.integration_info(ele_type)

    #  Initialize element stiffness array
    k_element = np.zeros((ndof * nelnodes, ndof * nelnodes))

    # Loop over integration points and add to k_element
    for int_pt in range(0, num_integration_points):
        # Local coordinates of integration point
        xi = xi_array[:, int_pt]
        # Shape function derivative
        dNdxi = di.shape_fcn_derivative(ele_type, xi)

        # Compute Jacobian and its inverse
        dxdxi, det_dxdxi = compute_jacobian(coords, dNdxi)
        dxidx = np.linalg.inv(dxdxi)
        dNdx = convert_derivatives(dNdxi, dxidx)

        # Compute deformation gradient F, left Cauchy-Green tensor, and Jacobian J
        F = compute_deformation_gradient(ncoord, displacement, dNdx)
        B = compute_B(F)
        J = compute_J(F)
        Finv = compute_Finv(F)

        # Convert shape function derivatives to spatial coordinates
        dNdxs = convert_to_spatial_derivatives(dNdx, Finv)

        # Compute stress and material stiffness
        stress = kirchhoff_stress(B, J, material_props)
        dsde = material_stiffness_2d(B, J, material_props)

        # Compute element stiffness matrix
        k_element += w_array[int_pt] * det_dxdxi * compute_stiffness_contributions(nelnodes, ndof, ncoord, stress, dsde, dNdxs)

    return k_element


def compute_jacobian(coords, dNdxi):
    """Computes the Jacobian matrix and its determinant."""
    dxdxi = coords @ dNdxi
    det_dxdxi = np.linalg.det(dxdxi)
    return dxdxi, det_dxdxi


def convert_derivatives(dNdxi, dxidx):
    """Converts shape function derivatives to global coordinates."""
    return dNdxi @ dxidx


def compute_B(F):
    """Compute the Left Cauchy-Green Tensor"""
    B = F @ F.T
    return B


def compute_J(F):
    """Compute the Jacobian determinant of the deformation gradient."""
    J = np.linalg.det(F)
    return J


def compute_Finv(F):
    """Compute the inverse of the deformation gradient."""
    Finv = np.linalg.inv(F)
    return Finv


def compute_deformation_gradient(ncoord, displacement, dNdx):
    """Computes the deformation gradient F."""
    F = np.eye(ncoord) + displacement @ dNdx
    return F


def convert_to_spatial_derivatives(dNdx, Finv):
    """Converts shape function derivatives to spatial coordinates."""
    return dNdx @ Finv


@njit
def material_stiffness_2d(B, J, materialprops):
    """
    Compute the 4th-order material stiffness tensor C_{ijkl} for plane strain
    in a 2D large deformation context.

    This function computes the 4th-order elasticity/stiffness tensor based on the isochoric
    (deviatoric) part of the deformation and the volumetric response.

    Parameters
    ----------
    B : numpy.ndarray (shape: (2, 2))
        The 2D left Cauchy-Green deformation tensor (or any symmetric tensor 
        representing the state of deformation in 2D).
    J : float
        The determinant of the deformation gradient (i.e., the volume change).
        Although J is strictly the volume ratio, for plane strain it is treated
        consistently in this 2D formulation.
    materialprops : array (length: 2)
        Material properties array containing:
          - mu1 (float): Shear modulus-like parameter.
          - K1 (float): Bulk modulus-like parameter.

    Returns
    -------
    C : numpy.ndarray (shape: (2, 2, 2, 2))
        The 4th-order material stiffness tensor for plane strain.

    Notes
    -----
    - Plane strain assumption implicitly treats the out-of-plane strain as zero,
      which leads to the additional +1 in the `Bqq` calculation here (accounting
      for the out-of-plane direction).
    - In nonlinear mechanics there are many equivalent ways to express the 
      consistent tangent
    """
    mu1, K1 = materialprops
    # Identity tensor in 2D
    dl = np.eye(2)

    # Trace of B for plane strain; plus 1.0 for the implicit out-of-plane direction
    Bqq = B[0, 0] + B[1, 1] + 1.0

    # Allocate the 4th-order stiffness tensor
    C = np.zeros((2, 2, 2, 2))

    # Compute each component
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    C[i, j, k, l] = (
                        mu1 * (
                            dl[i, k] * B[j, l]
                            + B[i, l] * dl[j, k]
                            - (2.0 / 3.0) * (
                                B[i, j] * dl[k, l]
                                + dl[i, j] * B[k, l]
                            )
                            + (2.0 / 3.0) * Bqq * dl[i, j] * dl[k, l] / 3.0
                        ) / (J ** (2.0 / 3.0))
                        + K1 * (2.0 * J - 1.0) * J * dl[i, j] * dl[k, l]
                    )
    return C


@njit
def kirchhoff_stress(B, J, materialprops):
    """
    Compute the Kirchhoff (or Cauchy-like) stress tensor given the left 
    Cauchy-Green deformation tensor B_{ij}, the determinant of the deformation 
    gradient J, and material properties.

    This function implements the logic of a common hyperelastic constitutive 
    relation in large deformations, returning the second Piola–Kirchhoff-like 
    stress for a compressible Neo-Hookean or related material model.

    Parameters
    ----------
    B : numpy.ndarray
        The left Cauchy-Green deformation tensor. For a 2D problem (plane strain), 
        `B` should be a 2x2 array. For a full 3D problem, `B` should be 3x3.
    J : float
        The determinant of the deformation gradient, i.e., the local volume 
        change ratio.
    materialprops : array_like (length: 2)
        Material properties array containing:
          - mu1 (float): Shear modulus-like parameter.
          - K1  (float): Bulk modulus-like parameter.

    Returns
    -------
    stress : numpy.ndarray
        Kirchhoff stress tensor. For a 2D input `B`, the output will be a 2x2 
        array; for 3D, a 3x3 array. The indices correspond to sigma_{ij}.

    Notes
    -----
    - For a 2D plane strain model, an out-of-plane stretch contribution is 
      implicitly accounted for by adding 1.0 to the trace `Bkk`. This is consistent
      with the standard plane strain assumption, where the out-of-plane direction
      is not free to deform.
    - For 3D, the trace is used as-is, with no additional term.
    """
    # Extract shear-like (mu1) and bulk-like (K1) parameters
    mu1, K1 = materialprops

    # Determine dimension from the shape of B
    ndof, ncoord = B.shape

    # Identity tensor
    dl = np.eye(ndof)

    # Trace of B
    Bkk = np.trace(B)

    # Adjustment for plane strain: add 1 if 2D
    if ndof == 2:
        Bkk += 1.0

    # Initialize the stress tensor
    stress = np.zeros((ndof, ncoord))

    # Compute stress components
    for i in range(ndof):
        for j in range(ncoord):
            # Deviatoric part (scaled by J^(-2/3)) plus volumetric part
            stress[i, j] = (
                mu1 * (B[i, j] - (Bkk / 3.0) * dl[i, j]) / (J ** (2.0 / 3.0))
                + K1 * J * (J - 1.0) * dl[i, j]
            )

    return stress


@njit
def compute_stiffness_contributions(nelnodes, ndof, ncoord, stress, dsde, dNdxs):
    """
    Computes contributions to the element stiffness matrix.

    Args:
        nelnodes (int): Number of nodes per element.
        ndof (int): Degrees of freedom per node.
        ncoord (int): Number of spatial coordinates (2D or 3D).
        stress (np.ndarray): Stress tensor (ndof x ncoord).
        dsde (np.ndarray): Material stiffness tensor (ndof x ncoord x ndof x ncoord).
        dNdxs (np.ndarray): Shape function derivatives in spatial coordinates (nelnodes x ncoord).

    Returns:
        np.ndarray: Element stiffness matrix (ndof * nelnodes, ndof * nelnodes).
    """
    num_dofs = ndof * nelnodes
    kel = np.zeros((num_dofs, num_dofs))

    for a in range(nelnodes):
        for b in range(nelnodes):
            for i in range(ndof):
                for k in range(ndof):
                    row = ndof * a + i
                    col = ndof * b + k

                    # Compute elastic contribution from material stiffness tensor (dsde)
                    matl_stiffness_term = 0.0
                    for j in range(ncoord):
                        for l in range(ncoord):
                            matl_stiffness_term += dsde[i, j, k, l] * dNdxs[b, l] * dNdxs[a, j]

                    # Compute stress-dependent contribution (geometric stiffness)
                    stress_term = 0.0
                    for j in range(ncoord):
                        # ALT -- old version -- stress_term += stress[i, j] * dNdxs[a, j] * dNdxs[b, k]
                        stress_term += stress[i, j] * dNdxs[a, k] * dNdxs[b, j]

                    # Update stiffness matrix
                    kel[row, col] += matl_stiffness_term - stress_term

    return kel
