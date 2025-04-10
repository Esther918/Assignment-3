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

L = 100.0      # length in x-direction
H = 20.0       # height in y-direction

# Stretch 100% extension
lambda_target = 2  
ndof = 2      
# [mu, K]
material_props = np.array([134.6, 83.33])  

# Generate mesh with circular arc end
def generate_arc_ends_mesh_2d(ele_type, x_min, y_min, L, H, nx, ny):
    coords = []
    connect = []
    R = H / 2  
    center_right = (x_min + L, y_min + H/2)  # Center of right arc

    # Generate nodes
    for i in range(nx + 1):
        x = x_min + i * L / nx
        for j in range(ny + 1):
            y = y_min + j * H / ny
            if i == nx:  # Right boundary
                angle = -np.pi/2 + (j * np.pi / ny)  
                x_new = center_right[0] + R * np.cos(angle)
                y_new = center_right[1] + R * np.sin(angle)
                coords.append([x_new, y_new])
            else:  # Interior nodes
                coords.append([x, y])

    coords = np.array(coords)

    # Generate connectivity
    if ele_type == "D2_nn3_tri": 
        for i in range(nx):
            for j in range(ny):
                n1 = i * (ny + 1) + j
                n2 = n1 + 1
                n3 = (i + 1) * (ny + 1) + j
                n4 = n3 + 1
                connect.append([n1, n3, n2])
                connect.append([n2, n3, n4])
    elif ele_type == "D2_nn6_tri":  
        coords_3node = coords.copy()
        coords, connect = add_midside_nodes(coords_3node, nx, ny)

    connect = np.array(connect)
    return coords, connect

# Function to add mid-side nodes for D2_nn6_tri
def add_midside_nodes(coords_3node, nx, ny):
    coords = coords_3node.tolist()
    connect = []
    node_offset = len(coords_3node)
    mid_node_map = {}

    for i in range(nx):
        for j in range(ny):
            n1 = i * (ny + 1) + j
            n2 = n1 + 1
            n3 = (i + 1) * (ny + 1) + j
            n4 = n3 + 1
            for (na, nb) in [(n1, n3), (n3, n2), (n2, n1)]: 
                key = tuple(sorted([na, nb]))
                if key not in mid_node_map:
                    mid_x = (coords_3node[na, 0] + coords_3node[nb, 0]) / 2
                    mid_y = (coords_3node[na, 1] + coords_3node[nb, 1]) / 2
                    mid_node_map[key] = node_offset
                    coords.append([mid_x, mid_y])
                    node_offset += 1
            for (na, nb) in [(n2, n3), (n3, n4), (n4, n2)]:
                key = tuple(sorted([na, nb]))
                if key not in mid_node_map:
                    mid_x = (coords_3node[na, 0] + coords_3node[nb, 0]) / 2
                    mid_y = (coords_3node[na, 1] + coords_3node[nb, 1]) / 2
                    mid_node_map[key] = node_offset
                    coords.append([mid_x, mid_y])
                    node_offset += 1
            n5 = mid_node_map[tuple(sorted([n1, n3]))]
            n6 = mid_node_map[tuple(sorted([n3, n2]))]
            n7 = mid_node_map[tuple(sorted([n2, n1]))]
            connect.append([n1, n3, n2, n5, n6, n7])
            n8 = mid_node_map[tuple(sorted([n2, n3]))]
            n9 = mid_node_map[tuple(sorted([n3, n4]))]
            n10 = mid_node_map[tuple(sorted([n4, n2]))]
            connect.append([n2, n3, n4, n8, n9, n10])

    return np.array(coords), np.array(connect)

# Identify boundaries
def identify_arc_boundaries(coords, connect, ele_type, x_min, x_max, y_min, y_max):
    boundary_nodes = {"left": [], "right": [], "top": [], "bottom": []}
    tol = 1e-6
    R = H / 2
    for i, (x, y) in enumerate(coords):
        if abs(y - y_min) < tol:
            boundary_nodes["bottom"].append(i)
        elif abs(y - y_max) < tol:
            boundary_nodes["top"].append(i)
        elif abs(x - x_min) < tol: 
            boundary_nodes["left"].append(i)
        elif abs(x - (x_max + R)) < R + tol: 
            boundary_nodes["right"].append(i)
    boundary_edges = pre.identify_rect_boundaries(coords, connect, ele_type, 0, L, 0, H)
    return boundary_nodes, boundary_edges

# Solver wrapper for FEA simulation
def run_fea_simulation(nx, ny, ele_type, label):
    coords, connect = generate_arc_ends_mesh_2d(ele_type, 0.0, 0.0, L, H, nx, ny)
    
    boundary_nodes, boundary_edges = identify_arc_boundaries(coords, connect, ele_type, 0, L, 0, H)
    fixed_left = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0.0, 0.0)
    fixed_right = pre.assign_fixed_nodes_rect(boundary_nodes, "right", (lambda_target - 1) * L, 0)
    fixed_top_y = pre.assign_fixed_nodes_rect(boundary_nodes, "top", None, 0.0)
    fixed_bottom_y = pre.assign_fixed_nodes_rect(boundary_nodes, "bottom", None, 0.0)
    fixed_nodes = np.hstack((fixed_left, fixed_right, fixed_top_y, fixed_bottom_y))

    dload_info = np.empty((ndof + 2, 0))
    nr_num_steps = 5

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
    tol_y = H / 20.0
    mid_nodes = [i for i in range(coords.shape[0]) if abs(coords[i, 1] - H/2) < tol_y]
    mid_nodes = sorted(mid_nodes, key=lambda i: coords[i, 0])

    x_vals = np.array([coords[i, 0] for i in mid_nodes])
    computed_u_x = np.array([final_disp[ndof * i] for i in mid_nodes])
    analytical_u_x = (lambda_target - 1) * x_vals

    # Calculate L2 error
    l2_error = np.sqrt(np.mean((computed_u_x - analytical_u_x)**2))
    return l2_error

print("h-refinement for FEA with D2_nn3_tri:")
h_refinement_steps = [(25, 4), (50, 8), (100, 16), (200, 32)]
h_errors = []
h_mesh_sizes = []
for nx, ny in h_refinement_steps:
    h = L / nx 
    error = run_fea_simulation(nx, ny, "D2_nn3_tri", label = "mesh size")
    h_errors.append(error)
    h_mesh_sizes.append(h)
    print(f" Mesh size: {h:.4f}, L2 error: {error:.6f}")

# Demonstrate p-refinement for FEA
p_refinement_steps = [(25, 4), (50, 8), (100, 16)]  
p_errors = []
p_mesh_sizes = []
for nx, ny in p_refinement_steps:
    print(f"Processing p-refinement: nx={nx}, ny={ny}, element type=D2_nn6_tri")
    h = L / nx
    error = run_fea_simulation(nx, ny, "D2_nn6_tri", label = "mesh size")
    p_errors.append(error)
    p_mesh_sizes.append(h)
    print(f" Mesh size: {h:.4f}, L2 error: {error:.6f}")

# Plot error vs mesh size
plt.figure(figsize=(10, 6))
plt.loglog(h_mesh_sizes, h_errors, 'bo-', label='h-refinement')
plt.loglog(p_mesh_sizes, p_errors, 'ro-', label='p-refinement')
plt.xlabel("Mesh Size (h)")
plt.ylabel("L2 Error in Displacement")
plt.title("Error vs Mesh Size: h-refinement vs p-refinement")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

img_fname = tutorials_dir / "error_vs_mesh_size_h_vs_p_refinement.png"
plt.savefig(str(img_fname))
