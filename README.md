# Assignment-3

## Run the test
```bash
conda create --name finite-element-analysis-env python=3.12.9
```

```bash
conda activate finite-element-analysis-env
```

```bash
python --version
```

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install -e .
```

```bash
pytest -v --cov=finiteelementanalysis --cov-report term-missing
```

## Tutorials
**Part A**
```bash
python partA.py
```
From full_code_example_2.py
* Beam geometry:  
L = 15.0   
H = 1.0    
nx = 60    
ny = 4     
ele_type = "D2_nn8_quad" 
ndof = 2
* Material properties:
E = 76000.0
nu = 0.3
*  Comparison to an analytical solution  
Numerical = -0.009110  
Analytical = -0.009093  
Absolute error = 1.732997e-05  

**Part B**
```bash
python partB.py
```
* Stretching 100% of a single layer of elastin network:  
L = 100.0     
H = 20.0  
BC: circular ends  
nx = 25      
ny = 4  
Note: see error_vs_mesh_size_h_vs_p_refinement.png for convergence with respect to mesh size.
![error](finiteelementanalysis/tutorials/error_vs_mesh_size_h_vs_p_refinement.png)  

**Part C**
```bash
python partC.py
```
* A very thin and long structure.
* The huge difference between shear modulus and bulk modulus worsens conditioning.
* Improper mesh size:  
d_displacement = spla.spsolve(K_sparse, R)
Iteration 1, Correction=0.000000e+00, Residual=nan, tolerance=1.000000e-08



## High-level overview schematic diagram of the code:
![workflow](workflow.jpg)

* pre_process: Generates mesh and labels boundaries.  
  Structured meshes: generate_rect_mesh_2d for quads and triangles.  
  Boundary Identification: identify_rect_boundaries labels nodes and faces on domain edges.  
* pre_process_demo_helper_fcns: Plot the mesh and field errors.  
  Mesh Visualization: plot_mesh_2D with nodes/elements/Gauss points.  
  Interpolation: interpolate_scalar_to_gauss_pts for nodal to Gauss point values.  
* local_element: Computes element stiffness and residual using shape functions from discretization.  
* assemble_global: Turns element matrices from local_element into global system.  
