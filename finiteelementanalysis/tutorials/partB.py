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

# FEA problem info
# h-refinement: ele_type = "D2_nn3_tri"
# p-refinement: ele_type = "D2_nn6_tri"
ele_type = "D2_nn6_tri"
ndof = 2

# Define domain
L = 100.0      # length in x-direction
H = 20.0        # height in y-direction
nx = 25        # number of elements in x
ny = 4         # number of elements in y

# Prescribed stretch 100% extension
lambda_target = 2

# Custom function to generate mesh with circular arc ends
def generate_arc_ends_mesh_2d(ele_type, x_min, y_min, L, H, nx, ny):
    # Assuming nx and ny define a grid, we'll adjust left and right boundaries
    coords = []
    connect = []

    # Radius of the circular ends (adjust as needed)
    R = H / 2  # Half height as radius for simplicity
    center_right = (x_min + L, y_min + H/2)  # Center of right arc

    # Generate nodes
    for i in range(nx + 1):
        x = x_min + i * L / nx
        for j in range(ny + 1):
            y = y_min + j * H / ny
            # Adjust x-coordinate for left and right ends to follow circular arc
            if i == nx:  # Right boundary
                angle = -np.pi/2 + (j * np.pi / ny)  # From -90° to 90°
                x_new = center_right[0] + R * np.cos(angle)
                y_new = center_right[1] + R * np.sin(angle)
                coords.append([x_new, y_new])
            else:  # Interior nodes (linear interpolation or original y)
                coords.append([x, y])

    coords = np.array(coords)

    # Generate connectivity
    if ele_type == "D2_nn3_tri":  # 3-node triangular elements
        for i in range(nx):
            for j in range(ny):
                n1 = i * (ny + 1) + j
                n2 = n1 + 1
                n3 = (i + 1) * (ny + 1) + j
                n4 = n3 + 1
                connect.append([n1, n3, n2])
                connect.append([n2, n3, n4])
    elif ele_type == "D2_nn6_tri":  # 6-node triangular elements
        # For p-refinement, we need mid-side nodes
        coords_3node = coords.copy()
        coords, connect = add_midside_nodes(coords_3node, nx, ny)

    connect = np.array(connect)
    return coords, connect

# Function to add mid-side nodes for 6-node triangles
def add_midside_nodes(coords_3node, nx, ny):
    coords = coords_3node.tolist()
    connect = []
    node_offset = len(coords_3node)
    mid_node_map = {}

    # Generate mid-side nodes
    for i in range(nx):
        for j in range(ny):
            n1 = i * (ny + 1) + j
            n2 = n1 + 1
            n3 = (i + 1) * (ny + 1) + j
            n4 = n3 + 1
            # Mid-side nodes (n5, n6, n7)
            for (na, nb) in [(n1, n3), (n3, n2), (n2, n1)]:  # First triangle
                key = tuple(sorted([na, nb]))
                if key not in mid_node_map:
                    mid_x = (coords_3node[na, 0] + coords_3node[nb, 0]) / 2
                    mid_y = (coords_3node[na, 1] + coords_3node[nb, 1]) / 2
                    mid_node_map[key] = node_offset
                    coords.append([mid_x, mid_y])
                    node_offset += 1
            # Second triangle mid-side nodes
            for (na, nb) in [(n2, n3), (n3, n4), (n4, n2)]:
                key = tuple(sorted([na, nb]))
                if key not in mid_node_map:
                    mid_x = (coords_3node[na, 0] + coords_3node[nb, 0]) / 2
                    mid_y = (coords_3node[na, 1] + coords_3node[nb, 1]) / 2
                    mid_node_map[key] = node_offset
                    coords.append([mid_x, mid_y])
                    node_offset += 1
            # Connectivity for first triangle (n1, n3, n2) with mid-nodes
            n5 = mid_node_map[tuple(sorted([n1, n3]))]
            n6 = mid_node_map[tuple(sorted([n3, n2]))]
            n7 = mid_node_map[tuple(sorted([n2, n1]))]
            connect.append([n1, n3, n2, n5, n6, n7])

            # Connectivity for second triangle (n2, n3, n4) with mid-nodes
            n8 = mid_node_map[tuple(sorted([n2, n3]))]
            n9 = mid_node_map[tuple(sorted([n3, n4]))]
            n10 = mid_node_map[tuple(sorted([n4, n2]))]
            connect.append([n2, n3, n4, n8, n9, n10])

    return np.array(coords), np.array(connect)

# Generate mesh with circular arc ends
coords, connect = generate_arc_ends_mesh_2d(ele_type, 0.0, 0.0, L, H, nx, ny)

mesh_img_fname = tutorials_dir / "partB_mesh.png"
pre_demo.plot_mesh_2D(str(mesh_img_fname), ele_type, coords, connect)

# Identify boundaries (adjusted for arc ends)
def identify_arc_boundaries(coords, connect, ele_type, x_min, x_max, y_min, y_max):
    boundary_nodes = {"left": [], "right": [], "top": [], "bottom": []}
    tol = 1e-6
    for i, (x, y) in enumerate(coords):
        if abs(y - y_min) < tol:
            boundary_nodes["bottom"].append(i)
        elif abs(y - y_max) < tol:
            boundary_nodes["top"].append(i)
        elif abs(x - x_min - H/2) < H/2 + tol and x <= x_min + H/2:  # Approximate left arc
            boundary_nodes["left"].append(i)
        elif abs(x - x_max + H/2) < H/2 + tol and x >= x_max - H/2:  # Approximate right arc
            boundary_nodes["right"].append(i)
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(coords, connect, ele_type, 0, L, 0, H)
    return boundary_nodes, boundary_edges

boundary_nodes, boundary_edges = identify_arc_boundaries(coords, connect, ele_type, 0, L, 0, H)

# Apply boundary conditions
fixed_left = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0.0, 0.0)
fixed_right = pre.assign_fixed_nodes_rect(boundary_nodes, "right", (lambda_target - 1) * L, 0)
fixed_top_y = pre.assign_fixed_nodes_rect(boundary_nodes, "top", None, 0.0)
fixed_bottom_y = pre.assign_fixed_nodes_rect(boundary_nodes, "bottom", None, 0.0)
fixed_nodes = np.hstack((fixed_left, fixed_right, fixed_top_y, fixed_bottom_y))

# No distributed load
dload_info = np.empty((ndof + 2, 0))

# Material properties
material_props = np.array([134.6, 83.33])  # [mu, K]

# Number of incremental loading steps
nr_num_steps = 5

# Run the solver
displacements_all, nr_info_all = hyperelastic_solver(
    material_props,
    ele_type,
    coords.T,
    connect.T,
    fixed_nodes,
    dload_info,
    nr_print=True,
    nr_num_steps=nr_num_steps,
    nr_tol=1e-8,
    nr_maxit=30,
)

final_disp = displacements_all[-1]

# Analytical solution comparison (midline)
tol_y = H / 20.0
mid_nodes = [i for i in range(coords.shape[0]) if abs(coords[i, 1] - H/2) < tol_y]
mid_nodes = sorted(mid_nodes, key=lambda i: coords[i, 0])

x_vals = np.array([coords[i, 0] for i in mid_nodes])
computed_u_x = np.array([final_disp[ndof * i] for i in mid_nodes])
analytical_u_x = (lambda_target - 1) * x_vals

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, computed_u_x, 'ro-', label="Computed u_x")
plt.plot(x_vals, analytical_u_x, 'b--', label="Analytical u_x")
plt.xlabel("x (m)")
plt.ylabel("u_x (m)")
plt.title("Comparison of u_x(x): Computed vs. Analytical")
plt.legend()
plt.grid(True)
plt.tight_layout()

img_fname = tutorials_dir / "uniaxial_extension_error.png"
plt.savefig(str(img_fname))

# Save animation
img_name = "partB_p_refinement.gif"
fname = str(tutorials_dir / img_name)
viz.make_deformation_gif(displacements_all, coords, connect, ele_type, fname)