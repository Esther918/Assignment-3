import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path


def plot_mesh_2D(fname: str, ele_type: str, coords: np.ndarray, connect: np.ndarray, displacements: np.ndarray, magnification: float = 1.0):
    """
    Plots the initial and deformed mesh for a 2D finite element model with displacement magnitude colormap.

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
    displacements : np.ndarray of shape (n_nodes, 2)
        The displacement vectors at each node.
    magnification : float, optional
        Factor to scale the visual displacement (default is 1.0).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Magnify the deformation for visualization
    coords_deformed = coords + magnification * displacements

    # Compute displacement magnitudes for colormap
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)

    for element in connect:
        element_coords = coords[element]
        element_coords_def = coords_deformed[element]

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

        # Plot initial mesh in light gray
        for edge in edges:
            ax.plot(element_coords[edge, 0], element_coords[edge, 1], color='lightgray', lw=1.0, alpha=0.8)

        # Plot deformed mesh in darker lines
        for edge in edges:
            ax.plot(element_coords_def[edge, 0], element_coords_def[edge, 1], color='black', lw=1.5)

    # Plot deformed nodes with color by displacement magnitude
    sc = ax.scatter(coords_deformed[:, 0], coords_deformed[:, 1], c=displacement_magnitudes,
                    cmap='coolwarm', s=20, edgecolors='k', linewidths=0.3, zorder=3)

    plt.colorbar(sc, ax=ax, label="Displacement Magnitude", fraction=0.046, pad=0.04)

    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"Initial and Deformed Mesh for {ele_type}", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return


def make_deformation_gif(displacements_all, coords, connect, ele_type, gif_path, magnification=1.0, interval=500):
    """
    Create an animated GIF showing the deformation progression of a 2D mesh with displacement magnitude colormap.

    Parameters
    ----------
    displacements_all : list of np.ndarray
        List of flattened displacement vectors, each of shape (n_nodes * 2,).
    coords : np.ndarray of shape (n_nodes, 2)
        The undeformed coordinates of the mesh nodes.
    connect : np.ndarray of shape (n_elements, nodes_per_element)
        Element connectivity.
    ele_type : str
        The element type string (e.g., "D2_nn4_quad").
    gif_path : str or Path
        Output file path for the animated GIF.
    magnification : float, optional
        Factor to scale the visual displacement (default = 1.0).
    interval : int, optional
        Time (in milliseconds) between frames in the GIF (default = 500).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set safe plot limits accounting for potential deformation
    disp_stack = np.stack([d.reshape(-1, 2) for d in displacements_all])
    max_disp = np.max(np.abs(disp_stack)) * magnification
    x_min, x_max = coords[:, 0].min() - max_disp, coords[:, 0].max() + max_disp
    y_min, y_max = coords[:, 1].min() - max_disp, coords[:, 1].max() + max_disp
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Compute global maximum displacement magnitude
    disp_mags = np.linalg.norm(disp_stack, axis=2)
    max_disp_mag = disp_mags.max()

    # Plot undeformed mesh in light gray
    for element in connect:
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

        element_coords = coords[element]
        for edge in edges:
            ax.plot(element_coords[edge, 0], element_coords[edge, 1], color='lightgray', lw=1.0, alpha=0.8)

    # Prepare animated objects
    mesh_lines = []
    for element in connect:
        for edge in edges:
            line, = ax.plot([], [], color='black', lw=1.5)
            mesh_lines.append((element[edge[0]], element[edge[1]], line))

    scatter = ax.scatter([], [], c=[], cmap='coolwarm', vmin=0, vmax=max_disp_mag,
                         s=20, edgecolors='k', linewidths=0.3, zorder=3)
    cbar = plt.colorbar(scatter, ax=ax, label="Displacement Magnitude", fraction=0.046, pad=0.04)

    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    title_text = ax.set_title("", fontsize=14, fontweight='bold')

    def update(frame_idx):
        disp_flat = displacements_all[frame_idx]
        disp = disp_flat.reshape(-1, 2)
        coords_def = coords + magnification * disp
        disp_mag = np.linalg.norm(disp, axis=1)

        for a, b, line in mesh_lines:
            line.set_data([coords_def[a, 0], coords_def[b, 0]], [coords_def[a, 1], coords_def[b, 1]])

        scatter.set_offsets(coords_def)
        scatter.set_array(disp_mag)

        title_text.set_text(f"{ele_type}  |  Frame {frame_idx + 1} of {len(displacements_all)}")
        return [line for _, _, line in mesh_lines] + [scatter, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(displacements_all), interval=interval, blit=True)
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(gif_path), writer='pillow')
    plt.close(fig)
    return
