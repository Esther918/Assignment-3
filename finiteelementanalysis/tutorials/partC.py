import warnings
warnings.simplefilter("always")
from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import pre_process_demo_helper_fcns as pre_demo
from finiteelementanalysis.solver import hyperelastic_solver
from finiteelementanalysis import visualize as viz
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# for saving files later
tutorials_dir = Path(__file__).parent

# Solve a homogeneous uniaxial extension problem and compare the computed displacement
# field to the analytical solution u_x = (lambda - 1)*x and u_y = 0.

# Boundary conditions:
#     - Left boundary (x=0): u = (0, 0)
#     - Right boundary (x=L): u = ((lambda - 1)*L, 0)
#     - Top and bottom boundaries (y = 0 and y = H): u_y = 0 (to enforce a homogeneous state)

# The analytical solution for a homogeneous deformation is then:
#     u_x(x) = (lambda - 1)*x,  u_y(x) = 0.

# FEA problem info
ele_type = "D2_nn3_tri"
ndof = 2

# Define domain
L = 100.0      # length in x-direction
H = 0.01       # height in y-direction
nx = 50       # number of elements in x
ny = 1       # number of elements in y, keep this an even number if you want the analytical solution to be able to compute midline deformation

# Prescribed stretch 100% extension
lambda_target = 2

# Generate mesh
coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, L, H, nx, ny)


mesh_img_fname = tutorials_dir / "partC_mesh.png"
pre_demo.plot_mesh_2D(str(mesh_img_fname), ele_type, coords, connect)


# Identify boundaries
boundary_nodes, boundary_edges = pre.identify_rect_boundaries(coords, connect, ele_type, 0, L, 0, H)

# Apply boundary conditions:
# 1. Fix left boundary: both u_x and u_y = 0.
fixed_left = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0.0, 0.0)
# 2. Prescribe right boundary: u_x = (lambda_target - 1)*L at x = L.
fixed_right = pre.assign_fixed_nodes_rect(boundary_nodes, "right", (lambda_target - 1) * L, 0)
# 3. To force a homogeneous state, prescribe u_y = 0 on the top and bottom boundaries.
fixed_top_y = pre.assign_fixed_nodes_rect(boundary_nodes, "top", None, 0.0)
fixed_bottom_y = pre.assign_fixed_nodes_rect(boundary_nodes, "bottom", None, 0.0)
# Combine BCs (assuming the functions return arrays of shape (3, n_bc))
fixed_nodes = np.hstack((fixed_left, fixed_right, fixed_top_y, fixed_bottom_y))

# No distributed load is applied
dload_info = np.empty((ndof + 2, 0))

# Choose material properties
material_props = np.array([1.346, 8333.0])  # [mu, K]

# Number of incremental loading steps
nr_num_steps = 3

# Run the solver
displacements_all, nr_info_all = hyperelastic_solver(
    material_props,
    ele_type,
    coords.T,      # solver expects coords as (ncoord, n_nodes)
    connect.T,     # and connectivity as (n_nodes_per_elem, n_elems)
    fixed_nodes,
    dload_info,
    nr_print=True,
    nr_num_steps=nr_num_steps,
    nr_tol=1e-8,
    nr_maxit=30,
)

final_disp = displacements_all[-1]  # final global displacement vector (length = n_nodes * ndof)

# Analytical solution: For a homogeneous extension,
#   u_x(x) = (lambda_target - 1) * x, and u_y(x) = 0.
# Extract nodes near mid-height to get a 1D slice.
tol_y = H / 20.0  # tolerance for y coordinate
mid_nodes = [i for i in range(coords.shape[0]) if abs(coords[i, 1] - H/2) < tol_y]
mid_nodes = sorted(mid_nodes, key=lambda i: coords[i, 0])  # sort by x-coordinate

# Extract x-coordinates and computed u_x from the final displacement.
x_vals = np.array([coords[i, 0] for i in mid_nodes])
computed_u_x = np.array([final_disp[ndof * i] for i in mid_nodes])
# Analytical solution: u_x(x) = (lambda_target - 1)*x.
analytical_u_x = (lambda_target - 1) * x_vals

# Plot the computed and analytical u_x vs. x.
plt.figure(figsize=(8, 6))
plt.plot(x_vals, computed_u_x, 'ro-', label="Computed u_x")
plt.plot(x_vals, analytical_u_x, 'b--', label="Analytical u_x")
plt.xlabel("x (m)")
plt.ylabel("u_x (m)")
plt.title("Comparison of u_x(x): Computed vs. Analytical")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot image to the tutorials directory.
img_fname = tutorials_dir / "uniaxial_extension_error.png"
plt.savefig(str(img_fname))

# Save an animation of the deformation
img_name = "partC.gif"
fname = str(tutorials_dir / img_name)
viz.make_deformation_gif(displacements_all, coords, connect, ele_type, fname)
