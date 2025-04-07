import numpy as np
from finiteelementanalysis import assemble_global as assemble
from finiteelementanalysis import discretization as di
import scipy.sparse.linalg as spla


def hyperelastic_solver(
    material_props,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    fixed_nodes: np.ndarray,
    dload_info: np.ndarray,
    nr_print: bool = False,
    nr_num_steps: int = 5,
    nr_tol: float = 1e-9,
    nr_maxit: int = 30
) -> list[np.ndarray]:
    """
    Solve a hyperelastic finite element problem using a Newton–Raphson scheme
    with incremental loading.

    This function computes the nodal displacements for a 2D or 3D hyperelastic
    model by incrementally applying external traction loads and iterating to
    equilibrium via the Newton–Raphson method.

    Parameters
    ----------
    material_props : array_like
        Material properties needed by the element stiffness and residual routines
        (e.g. [mu, bulk_modulus] for a Neo-Hookean model).
    ele_type : str
        The element type identifier (e.g. 'D2_nn4_quad', 'D3_nn8_hex'). Used to
        determine the spatial dimension (ncoord), the DOFs per node (ndof),
        and the typical number of nodes per element.
    coords : numpy.ndarray, shape (ncoord, n_nodes)
        The global coordinates of all nodes in the mesh. For 2D, coords might
        have shape (2, n_nodes); for 3D, shape (3, n_nodes).
    connect : numpy.ndarray, shape (max_nodes_per_elem, n_elems)
        The connectivity array. connect[a, e] gives the global node index
        of the a-th local node of element e. Assumed 0-based indexing.
    fixed_nodes : numpy.ndarray, shape (3, n_fixed)
        Prescribed boundary conditions. Each column contains:
          - fixed_nodes[0, j]: The global node index where the boundary condition is applied.
          - fixed_nodes[1, j]: The DOF index (within that node) being fixed.
          - fixed_nodes[2, j]: The displacement value to be applied.
        (Adjust if your indexing differs.)
    dload_info : numpy.ndarray, shape (ndof+2, n_face_loads)
        An array describing distributed face loads, where each column corresponds
        to one face load specification:
          - dload_info[0, j]: The element index (0-based) containing the face.
          - dload_info[1, j]: The face identifier (e.g., 0..5 for a hex).
          - dload_info[i+2, j] for i in [0..(ndof-1)]: The traction components
            on that face (e.g., tx, ty, [tz] if 3D).
    nr_print : bool, optional (default=False)
        If True, print iteration details (Newton–Raphson steps, residual norms, etc.).
    nr_num_steps : int, optional (default=5)
        The number of incremental load steps to apply.
    nr_tol : float, optional (default=1e-9)
        Convergence tolerance for the Newton–Raphson iteration, based on the 
        displacement correction norm.
    nr_maxit : int, optional (default=30)
        The maximum number of Newton–Raphson iterations per load step.

    Returns
    -------
    displacements_all : list of numpy.ndarray
        A list of length `nr_num_steps`, where each entry is the global
        displacement vector (of length n_nodes * ndof) at the end of that
        load increment. The final entry in this list is the converged solution
        after the last load step.

    Notes
    -----
    - The main assembly routines for global stiffness, traction, and residual 
      (`assemble.global_stiffness`, `assemble.global_traction`, and 
      `assemble.global_residual`) are called here. They must be defined 
      elsewhere or imported from a module.
    - Displacement boundary conditions are applied by modifying the global
      stiffness matrix and residual vector. Each fixed DOF row is zeroed 
      in the stiffness, set to 1 on the diagonal, and forced in the residual 
      to match the prescribed displacement value.
    - This function uses a standard Newton–Raphson iteration for each 
      incremental load step, updating the displacement until the correction 
      norm satisfies `nr_tol` or the iteration count reaches `nr_maxit`.
    - The variable `wnorm` measures the norm of the updated displacement vector, 
      and `err1` is based on the norm of the correction `d_displacement`.
    """
    nnode = coords.shape[1]
    _, ndof, _ = di.element_info(ele_type)
    num_fixed_dofs = fixed_nodes.shape[1]

    # initialize displacement vector of length (nnode * ndof)
    displacements_all = []
    displacement = np.zeros(nnode * ndof)

    # vector to store print values
    nr_print_info_all = []

    # solver loop, outer loop applies load incrementally
    for step in range(nr_num_steps):
        nr_print_info = []
        loadfactor = (step + 1) / nr_num_steps
        err1 = 1.0
        nit = 0
        if nr_print:
            print("Step %i, load factor = %0.3f" % (step, loadfactor))

        while (err1 > nr_tol) and (nit < nr_maxit):
            nit += 1

            # Use ndof from element info for reshaping
            displacement_reshaped = displacement.reshape(-1, ndof).T

            # Assemble global matrices/vectors
            K = assemble.global_stiffness(ele_type, coords, connect, material_props, displacement_reshaped)
            F = assemble.global_traction(ele_type, coords, connect, dload_info)
            R = loadfactor * F - assemble.global_residual(ele_type, coords, connect, material_props, displacement_reshaped)

            # Apply prescribed displacement BCs
            for n in range(num_fixed_dofs):
                node_id = fixed_nodes[0, n]
                dof_id = fixed_nodes[1, n]
                bc_dof = int(ndof * node_id + dof_id)
                K[bc_dof, :] = 0.0
                K[bc_dof, bc_dof] = 1.0
                R[bc_dof] = loadfactor * fixed_nodes[2, n] - displacement[bc_dof]

            d_displacement = np.linalg.solve(K, R)
            displacement += d_displacement 

            wnorm = np.dot(displacement, displacement)
            err1 = np.dot(d_displacement, d_displacement)
            err2 = np.dot(R, R)
            err1 = np.sqrt(err1 / wnorm) if wnorm > 1e-16 else 0.0
            err2 = np.sqrt(err2) / (ndof * nnode)

            nr_str = "Iteration %i, Correction=%0.6e, Residual=%0.6e, tolerance=%0.6e" % (nit, err1, err2, nr_tol)
            nr_print_info.append(nr_str)
            if nr_print:
                print(nr_str)

        # Append a copy of the displacement for this load step
        displacements_all.append(displacement.copy())
        nr_print_info_all.append(nr_print_info.copy())

    return displacements_all, nr_print_info_all