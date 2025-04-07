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
ny = 2     
ele_type = "D2_nn8_quad" 
ndof = 2
* Material properties:
E = 76000.0
nu = 0.3

**Part B**
```bash
python partB.py
```
* Stretching 100% of a single layer of elastin network:

L = 100.0     
H = 10.0

BC: circular ends

nx = 50      
ny = 5           
ele_type = "D2_nn3_tri"
ndof = 2

**Part C**
```bash
python partC.py
```
Creating an ill conditioned K
* A very thinand long structure.
* The huge difference between shear modulus and bulk modulus worsens conditioning.
* Improper mesh size:  
d_displacement = spla.spsolve(K_sparse, R)
Iteration 1, Correction=0.000000e+00, Residual=nan, tolerance=1.000000e-08



## High-level overview schematic diagram of the code:
![overview](overview.jpg)

* pre_process: Generates mesh and labels boundaries.
* pre_process_demo_helper_fcns: Plot the mesh.
* local_element: Computes element stiffness and residual using shape functions from discretization.
* assemble_global: Turns element matrices from local_element into global system.
