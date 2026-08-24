# FEM2D-Heat-Conduction
# FEM2D-Heat-Conduction

**2D Finite Element Solver for Steady-State Heat Conduction from Scratch**

## Overview

Pure Python implementation of the Galerkin Finite Element Method (FEM) for solving 2D steady-state heat conduction on a square plate with a circular hole. Uses **linear triangular elements (T3)** with analytical stiffness matrix integration.

## Features

- **Mesh generation:** Structured triangular mesh with circular hole removal
- **Element stiffness matrix:** Analytical integration for linear triangles
- **Global assembly:** Direct stiffness method
- **Boundary conditions:** Dirichlet (penalty method) + Neumann (natural)
- **Post-processing:** Contour plots, 3D surface, heat flux computation
- **Verification:** h-refinement convergence study with error analysis

## Physics

Equation: `-d/dx(k dT/dx) - d/dy(k dT/dy) = Q`

Boundary conditions:
- Left edge: `T = 100` (hot wall)
- Right edge: `T = 0` (cold wall)
- Top/Bottom/Hole: `dT/dn = 0` (insulated)

## Installation

```bash
pip install numpy matplotlib
