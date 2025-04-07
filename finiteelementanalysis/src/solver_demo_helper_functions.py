import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from finiteelementanalysis import assemble_global as assemble
from finiteelementanalysis import discretization as di
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import timeit
from typing import Literal


def compute_bandwidth(K):
    """
    Computes the (upper) bandwidth of matrix K.
    Bandwidth is the maximum |i - j| such that K[i, j] != 0.
    
    Parameters
    ----------
    K : np.ndarray or scipy.sparse matrix
        The stiffness matrix.
    
    Returns
    -------
    bandwidth : int
        The (upper) bandwidth of the matrix.
    """
    if sp.issparse(K):  # K is already in sparse format
        K = K.tocoo()  # Convert to COOrdinate format to access row/col indices
        if K.nnz == 0:  # max will fail for all 0s
            return 0
        bandwidth = np.max(np.abs(K.row - K.col))
    else:  # K is in standard non-sparse format
        row_inds, col_inds = np.nonzero(K)
        if row_inds.size == 0:  # max will fail for all 0s
            return 0
        bandwidth = np.max(np.abs(row_inds - col_inds))
    return bandwidth


def compute_condition_number(K):
    """
    Computes the condition number of matrix K.

    A large condition number indicates that the matrix is nearly singular or
    ill-conditioned, which can lead to significant numerical errors when solving
    linear systems. In particular, the condition number gives a bound on
    how much relative error in the input (e.g., round-off or discretization error)
    can be amplified in the solution.

    For example, if a matrix has a condition number of 1e10, then up to 10 digits
    of precision can be lost in the solution due to round-off error, even if the
    input data is exact. In finite element simulations, this can result in
    inaccurate displacements, stresses, or energy estimates, and it may also cause
    iterative solvers to converge slowly or fail entirely.

    Monitoring the condition number helps assess the stability and reliability
    of a numerical solution, especially when using fine meshes, high aspect ratio
    elements, or nearly incompressible materials.

    For more background on how numerical precision and round-off errors arise,
    see references on floating point arithmetic such as IEEE 754, the book
    "What Every Computer Scientist Should Know About Floating-Point Arithmetic,"
    or similar.

    For dense matrices, this function uses `np.linalg.cond`, which performs
    a full singular value decomposition (SVD) to compute the exact condition
    number as the ratio of the largest to smallest singular values.

    For sparse matrices, the condition number is estimated using partial
    SVD (`scipy.sparse.linalg.svds`) or eigenvalue methods. This is because
    computing the full SVD of large sparse matrices is prohibitively expensive
    in both time and memory.

    In practice, estimating the condition number from a small number of
    singular values is sufficient to:
    - Assess how ill-conditioned the matrix is
    - Evaluate solver stability and expected numerical accuracy
    - Guide preconditioner design in iterative methods

    Parameters
    ----------
    K : np.ndarray or scipy.sparse matrix

    Returns
    -------
    cond_num : float
        Estimated condition number. Returns np.inf if matrix is singular
        or estimate fails.
    """
    try:
        if sp.issparse(K):
            # Use sparse SVD to estimate condition number
            _, s, _ = spla.svds(K, k=2, return_singular_vectors=True)
            cond_num = max(s) / min(s) if min(s) > 0 else np.inf
        else:
            cond_num = np.linalg.cond(K)
    except Exception:
        cond_num = np.inf  # If any error occurs, treat as ill-conditioned

    return cond_num


def compute_sparsity(K):
    """
    Computes the sparsity of matrix K.

    Sparsity is defined as the percentage of zero entries in the matrix:
        sparsity = 100 * (# of zero entries) / (total # of entries)

    Parameters
    ----------
    K : np.ndarray or scipy.sparse matrix
        The matrix whose sparsity is to be computed.

    Returns
    -------
    sparsity : float
        Sparsity percentage (0 to 100). A higher value means a sparser matrix.
    """
    total_entries = K.shape[0] * K.shape[1]

    if sp.issparse(K):
        nonzero = K.nnz
    else:
        nonzero = np.count_nonzero(K)

    zero_entries = total_entries - nonzero
    sparsity = 100 * zero_entries / total_entries

    return sparsity


def analyze_and_visualize_matrix(K, fname: str, method: str = "imshow"):
    """
    Computes matrix diagnostics (condition number, sparsity, bandwidth) and 
    visualizes the sparsity pattern of the matrix.

    Parameters
    ----------
    K : np.ndarray or scipy.sparse matrix
        The matrix to analyze and visualize.
    fname : str
        The file path to save the visualization.
    method : str, optional
        Visualization method: "imshow" (default, uses colormap) or "spy"
        (scatter-style for sparse matrices, black dots).
    """
    # Convert to sparse format if not already
    if not sp.issparse(K):
        K = sp.csr_matrix(K)

    # Compute metrics
    cond_num = compute_condition_number(K)
    sparsity = compute_sparsity(K)
    bandwidth = compute_bandwidth(K)

    # Create plot
    plt.figure(figsize=(8, 8))

    if method == "imshow":
        plt.imshow(K.toarray() != 0, cmap="cividis", interpolation="none", aspect="equal")
    elif method == "spy":
        plt.spy(K, markersize=1, color="black")
    else:
        raise ValueError(f"Unsupported visualization method: '{method}'. Use 'imshow' or 'spy'.")

    # Title and formatting
    plt.title(
        f"Sparsity pattern of stiffness matrix\n"
        f"Condition number: {cond_num:.2e}\n"
        f"Sparsity: {sparsity:.2f}% | Bandwidth: {bandwidth}"
    )
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return


def visualize_stiffness_matrix(
    fname: str,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: np.ndarray,
    displacement: np.ndarray,
    method: str = "imshow"
):
    """
    Assembles the global stiffness matrix and visualizes its sparsity pattern.

    Parameters
    ----------
    fname : str
        The name of the file to store the plot as.
    ele_type : str
        The type of finite element.
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    coords : np.ndarray
        Global coordinates of all nodes, shape (ncoord, nnode).
    connect : np.ndarray
        Element connectivity array, shape (nelnodes, nelem).
    material_props : list or np.ndarray
        Material properties, e.g., [mu, bulk_modulus].
    displacement : np.ndarray
        Global displacement vector, shape (ndof, nnode).
    method : str, optional
        Visualization method: "imshow" (default, color-mapped) or "spy"
        (marker-based sparse visualization using black dots).
    """
    # Assemble global stiffness matrix
    K = assemble.global_stiffness(ele_type, coords, connect, material_props, displacement)

    # Analyze and visualize the matrix
    analyze_and_visualize_matrix(K, fname, method=method)
    return


def time_function_call(func, *args, num_runs=10, **kwargs):
    """
    Times the average execution time of a function call using timeit.

    Parameters
    ----------
    func : callable
        The function to time.
    *args : tuple
        Positional arguments to pass to the function.
    num_runs : int, optional
        Number of runs to average over (default: 10).
    **kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    avg_time : float
        Average execution time in seconds.
    """
    def wrapper():
        func(*args, **kwargs)

    timer = timeit.Timer(wrapper)
    total_time = timer.timeit(number=num_runs)
    avg_time = total_time / num_runs

    return avg_time


def time_assemble_global_stiffness(
    num_runs: int,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: np.ndarray,
    displacement: np.ndarray
) -> float:
    """
    Times the average execution time of the global stiffness matrix assembly.

    This function repeatedly assembles the global stiffness matrix for a given
    finite element mesh and material configuration, and returns the average time
    taken per assembly using the time_function_call utility.

    Parameters
    ----------
    num_runs : int
        Number of times to run the assembly for averaging.
    ele_type : str
        The type of finite element. Options include:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    coords : np.ndarray
        Global coordinates of all nodes, shape (ncoord, nnode).
    connect : np.ndarray
        Element connectivity array, shape (nelnodes, nelem).
    material_props : list or np.ndarray
        Material properties, e.g., [mu, bulk_modulus].
    displacement : np.ndarray
        Global displacement vector, shape (ndof, nnode).

    Returns
    -------
    avg_time : float
        Average time (in seconds) to assemble the global stiffness matrix over
        `num_runs` executions.
    """
    avg_time: float = time_function_call(
        assemble.global_stiffness,
        ele_type,
        coords,
        connect,
        material_props,
        displacement,
        num_runs=num_runs
    )

    return avg_time


def time_assemble_global_stiffness_sparse(
    num_runs: int,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: np.ndarray,
    displacement: np.ndarray
) -> float:
    """
    Times the average execution time of the global stiffness matrix assembly.

    This function repeatedly assembles the global stiffness matrix for a given
    finite element mesh and material configuration, and returns the average time
    taken per assembly using the time_function_call utility.

    Parameters
    ----------
    num_runs : int
        Number of times to run the assembly for averaging.
    ele_type : str
        The type of finite element. Options include:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    coords : np.ndarray
        Global coordinates of all nodes, shape (ncoord, nnode).
    connect : np.ndarray
        Element connectivity array, shape (nelnodes, nelem).
    material_props : list or np.ndarray
        Material properties, e.g., [mu, bulk_modulus].
    displacement : np.ndarray
        Global displacement vector, shape (ndof, nnode).

    Returns
    -------
    avg_time : float
        Average time (in seconds) to assemble the global stiffness matrix over
        `num_runs` executions.
    """
    avg_time: float = time_function_call(
        assemble.global_stiffness_sparse,
        ele_type,
        coords,
        connect,
        material_props,
        displacement,
        num_runs=num_runs
    )

    return avg_time


def time_assemble_global_traction(
    num_runs: int,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    dload_info: np.ndarray
) -> float:
    """
    Times the average execution time of the global traction vector assembly.

    This function repeatedly assembles the global traction vector for a given
    finite element mesh and distributed load configuration, and returns the
    average time taken per assembly using the time_function_call utility.

    Parameters
    ----------
    num_runs : int
        Number of times to run the assembly for averaging.
    ele_type : str
        The type of finite element. Options include:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    coords : np.ndarray
        Global coordinates of all nodes, shape (ncoord, nnode).
    connect : np.ndarray
        Element connectivity array, shape (nelnodes, nelem).
    dload_info : np.ndarray
        Array describing distributed face loads, shape (ndof+2, n_face_loads).

    Returns
    -------
    avg_time : float
        Average time (in seconds) to assemble the global traction vector over
        `num_runs` executions.
    """
    avg_time: float = time_function_call(
        assemble.global_traction,
        ele_type,
        coords,
        connect,
        dload_info,
        num_runs=num_runs
    )

    return avg_time


def time_assemble_global_residual(
    num_runs: int,
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: np.ndarray,
    displacement: np.ndarray
) -> float:
    """
    Times the average execution time of the global residual vector assembly.

    This function repeatedly assembles the global residual vector for a given
    finite element mesh and material configuration, and returns the average time
    taken per assembly using the time_function_call utility.

    Parameters
    ----------
    num_runs : int
        Number of times to run the assembly for averaging.
    ele_type : str
        The type of finite element. Options include:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    coords : np.ndarray
        Global coordinates of all nodes, shape (ncoord, nnode).
    connect : np.ndarray
        Element connectivity array, shape (nelnodes, nelem).
    material_props : list or np.ndarray
        Material properties, e.g., [mu, bulk_modulus].
    displacement : np.ndarray
        Global displacement vector, shape (ndof, nnode).

    Returns
    -------
    avg_time : float
        Average time (in seconds) to assemble the global stiffness matrix over
        `num_runs` executions.
    """
    avg_time: float = time_function_call(
        assemble.global_residual,
        ele_type,
        coords,
        connect,
        material_props,
        displacement,
        num_runs=num_runs
    )

    return avg_time


def prep_for_matrix_solve(
    ele_type: str,
    coords: np.ndarray,
    connect: np.ndarray,
    material_props: np.ndarray,
    displacement: np.ndarray,
    fixed_nodes: np.ndarray,
    dload_info: np.ndarray
):
    """
    Given:
    ele_type : str
        The type of finite element. Options include:
        - "D2_nn3_tri" : 3-node linear triangular element
        - "D2_nn6_tri" : 6-node quadratic triangular element
        - "D2_nn4_quad" : 4-node bilinear quadrilateral element
        - "D2_nn8_quad" : 8-node quadratic quadrilateral element
    coords : np.ndarray
        Global coordinates of all nodes, shape (ncoord, nnode).
    connect : np.ndarray
        Element connectivity array, shape (nelnodes, nelem).
    material_props : list or np.ndarray
        Material properties, e.g., [mu, bulk_modulus].
    displacement : np.ndarray
        Global displacement vector, shape (ndof, nnode).
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

    Returns:
    K : global stiffness matrix
    F : global total residual vector
    At one snapshot for the purpose of assessing matrix solver performance in a different function.
    """
    # Assemble, K, F, and R
    K = assemble.global_stiffness(ele_type, coords, connect, material_props, displacement)
    F = assemble.global_traction(ele_type, coords, connect, dload_info)
    R = F - assemble.global_residual(ele_type, coords, connect, material_props, displacement)

    displacement = displacement.T.reshape(-1)

    # Apply prescribed displacement BCs
    num_fixed_dofs = fixed_nodes.shape[1]
    _, ndof, _ = di.element_info(ele_type)
    for n in range(num_fixed_dofs):
        node_id = fixed_nodes[0, n]
        dof_id = fixed_nodes[1, n]
        bc_dof = int(ndof * node_id + dof_id)
        K[bc_dof, :] = 0.0
        K[bc_dof, bc_dof] = 1.0
        R[bc_dof] = fixed_nodes[2, n] - displacement[bc_dof]

    return K, R


def time_one_matrix_solve(
    K: np.ndarray,
    R: np.ndarray,
    method: Literal["dense", "sparse", "sparse_iterative"],
    num_runs: int = 3
) -> float:
    """
    Times a single matrix solve operation using the specified method.

    Parameters
    ----------
    K : np.ndarray
        Global stiffness matrix (dense or convertible to sparse).
    R : np.ndarray
        Global residual vector (same shape as K.shape[0]).
    method : str
        One of:
        - "dense": uses np.linalg.solve (dense matrix).
        - "sparse": uses scipy.sparse.linalg.spsolve (after converting K to sparse).
        - "sparse_iterative": uses gmres with ILU preconditioner.
    num_runs : int
        Number of runs to average over for timing (default: 3).

    Returns
    -------
    avg_time : float
        Average time in seconds for the solve operation.
    """

    if method == "dense":
        func = np.linalg.solve
        args = (K, R)

    elif method == "sparse":
        K_sparse = sp.csr_matrix(K)
        func = spla.spsolve
        args = (K_sparse, R)

    elif method == "sparse_iterative":
        K_sparse = sp.csr_matrix(K)
        try:
            # Incomplete LU for preconditioning
            ilu = spla.spilu(K_sparse.tocsc())
            M = spla.LinearOperator(K_sparse.shape, ilu.solve)
        except RuntimeError:
            # ILU factorization failed — fallback to no preconditioner
            M = None

        def gmres_solve(K_in, R_in):
            x, info = spla.gmres(K_in, R_in, M=M, atol=1e-8)
            if info != 0:
                raise RuntimeError(f"GMRES failed to converge (info={info})")
            return x

        func = gmres_solve
        args = (K_sparse, R)

    else:
        raise ValueError(f"Unsupported method: '{method}'")

    # Time the chosen solver
    avg_time = time_function_call(func, *args, num_runs=num_runs)

    return avg_time
