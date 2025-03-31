# Assignment-3

High-level overview schematic diagram of the code:
![overview](overview.jpg)

* pre_process: Generates mesh and labels boundaries.
* pre_process_demo_helper_fcns: Plot the mesh.
* local_element: Computes element stiffness and residual using shape functions from discretization.
* assemble_global: Turns element matrices from local_element into global system.
