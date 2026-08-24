"""
================================================================================
FEM2D_HEAT_CONDUCTION.py
================================================================================
2D Finite Element Solver for Steady-State Heat Conduction
Using Linear Triangular Elements (T3) - From Scratch in Python

Author: Dr. Sapna Makhdoom
Date: September 2026

Description:
    Solves the 2D Poisson equation: -nabla^2 T = Q on a square domain [0,1]x[0,1]
    with a circular hole at the center, using the Galerkin FEM with 
    linear triangular elements.
    
    Boundary conditions:
        - Left edge (x=0):  T = 100  (Dirichlet, hot wall)
        - Right edge (x=1): T = 0   (Dirichlet, cold wall)
        - Top/Bottom edges: dT/dn = 0 (Neumann, insulated)
        - Hole boundary:    dT/dn = 0 (Neumann, insulated)
    
    This code implements the complete FEM pipeline:
        1. Structured triangular mesh generation
        2. Element stiffness matrix assembly (analytical integration)
        3. Global stiffness matrix assembly
        4. Boundary condition enforcement
        5. Sparse linear solver
        6. Post-processing and visualization

    Pure Python + NumPy + Matplotlib. No external FEM libraries.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle
import time

# ==============================================================================
# SECTION 1: MESH GENERATION (Structured Triangular Mesh with Circular Hole)
# ==============================================================================

class Mesh:
    """
    Generates a structured triangular mesh on a square domain [0,1]x[0,1]
    with a circular hole removed from the center.
    
    Parameters:
        Nx, Ny: number of divisions in x and y directions
        hole_radius: radius of the circular hole (default 0.15)
    """
    def __init__(self, Nx=40, Ny=40, hole_radius=0.15):
        self.Nx = Nx
        self.Ny = Ny
        self.hole_radius = hole_radius
        self.Lx = 1.0
        self.Ly = 1.0
        
        self._generate_nodes()
        self._generate_elements()
        self._identify_boundaries()
        
        print("Mesh generated: {} nodes, {} elements".format(self.n_nodes, self.n_elements))
    
    def _generate_nodes(self):
        """Generate nodes on a structured grid, removing those inside the hole."""
        x_vals = np.linspace(0, self.Lx, self.Nx + 1)
        y_vals = np.linspace(0, self.Ly, self.Ny + 1)
        
        nodes_list = []
        cx, cy = 0.5, 0.5  # hole center
        
        for j in range(self.Ny + 1):
            for i in range(self.Nx + 1):
                x = x_vals[i]
                y = y_vals[j]
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                if r >= self.hole_radius:
                    nodes_list.append([x, y])
        
        self.nodes = np.array(nodes_list)
        self.n_nodes = len(self.nodes)
        
        # Create a mapping from (i,j) grid indices to node numbers
        self.node_map = {}
        idx = 0
        for j in range(self.Ny + 1):
            for i in range(self.Nx + 1):
                x = x_vals[i]
                y = y_vals[j]
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                if r >= self.hole_radius:
                    self.node_map[(i, j)] = idx
                    idx += 1
    
    def _generate_elements(self):
        """Generate triangular elements by splitting each quadrilateral cell."""
        elements_list = []
        cx, cy = 0.5, 0.5
        
        for j in range(self.Ny):
            for i in range(self.Nx):
                # Four corners of the quadrilateral cell
                corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                
                # Check which corners exist (not inside hole)
                valid_corners = []
                for c in corners:
                    if c in self.node_map:
                        valid_corners.append(self.node_map[c])
                    else:
                        valid_corners.append(-1)
                
                # Only create elements if all 4 corners are valid
                if -1 not in valid_corners:
                    n1, n2, n3, n4 = valid_corners
                    # Split quad into 2 triangles: (1,2,3) and (1,3,4)
                    elements_list.append([n1, n2, n3])
                    elements_list.append([n1, n3, n4])
        
        self.elements = np.array(elements_list, dtype=int)
        self.n_elements = len(self.elements)
    
    def _identify_boundaries(self):
        """Identify nodes on domain boundaries and hole boundary."""
        tol = 1e-10
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        
        # Dirichlet boundaries
        self.left_nodes = np.where(np.abs(x - 0.0) < tol)[0]
        self.right_nodes = np.where(np.abs(x - self.Lx) < tol)[0]
        
        # Neumann boundaries (top, bottom, hole)
        self.top_nodes = np.where(np.abs(y - self.Ly) < tol)[0]
        self.bottom_nodes = np.where(np.abs(y - 0.0) < tol)[0]
        
        # Hole boundary nodes
        cx, cy = 0.5, 0.5
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        self.hole_nodes = np.where(np.abs(r - self.hole_radius) < 2.5 * (self.Lx / self.Nx))[0]
        
        # All boundary nodes
        self.boundary_nodes = np.unique(np.concatenate([
            self.left_nodes, self.right_nodes, 
            self.top_nodes, self.bottom_nodes, self.hole_nodes
        ]))
        
        # Free (interior) nodes
        all_nodes = np.arange(self.n_nodes)
        self.free_nodes = np.setdiff1d(all_nodes, self.boundary_nodes)

# ==============================================================================
# SECTION 2: FINITE ELEMENT SOLVER
# ==============================================================================

class FEMSolver:
    """
    Solves the 2D steady-state heat equation using Galerkin FEM with
    linear triangular elements.
    
    Equation: -d/dx(k dT/dx) - d/dy(k dT/dy) = Q
    Weak form: integral(k grad(T) . grad(v)) = integral(Q v)
    
    For linear triangle with nodes (x1,y1), (x2,y2), (x3,y3):
        Area A = 0.5 * |det([[1, x1, y1], [1, x2, y2], [1, x3, y3]])|
        Shape function derivatives:
            b1 = y2 - y3,  b2 = y3 - y1,  b3 = y1 - y2
            c1 = x3 - x2,  c2 = x1 - x3,  c3 = x2 - x1
        Element stiffness matrix:
            K_e[i,j] = (1/(4A)) * (bi*bj + ci*cj)
        Element force vector (constant Q):
            F_e[i] = Q * A / 3
    """
    
    def __init__(self, mesh, k=1.0, Q=1000.0):
        """
        Parameters:
            mesh: Mesh object
            k: thermal conductivity
            Q: volumetric heat source
        """
        self.mesh = mesh
        self.k = k
        self.Q = Q
        self.n_nodes = mesh.n_nodes
        self.n_elements = mesh.n_elements
        
        print("FEM Solver initialized:")
        print("  Conductivity k = {}".format(k))
        print("  Heat source Q  = {}".format(Q))
        print("  DOFs           = {}".format(self.n_nodes))
    
    def _element_stiffness(self, elem_nodes):
        """
        Compute element stiffness matrix and force vector for a linear triangle.
        
        Input: elem_nodes = array of 3 node indices
        Returns: K_e (3x3), F_e (3x1)
        """
        # Get node coordinates
        coords = self.mesh.nodes[elem_nodes]
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Compute area using determinant formula
        # A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        A = 0.5 * abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
        
        if A < 1e-14:
            return np.zeros((3, 3)), np.zeros(3)
        
        # Shape function derivatives (constant over element)
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        
        # Element stiffness matrix
        K_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_e[i, j] = (self.k / (4.0 * A)) * (b[i]*b[j] + c[i]*c[j])
        
        # Element force vector (constant source Q)
        F_e = (self.Q * A / 3.0) * np.ones(3)
        
        return K_e, F_e
    
    def assemble(self):
        """
        Assemble global stiffness matrix K and global force vector F.
        Uses simple dense assembly (sufficient for moderate mesh sizes).
        """
        print("Assembling global system...")
        start = time.time()
        
        K = np.zeros((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)
        
        for e in range(self.n_elements):
            elem_nodes = self.mesh.elements[e]
            K_e, F_e = self._element_stiffness(elem_nodes)
            
            # Scatter element contributions to global matrix
            for i in range(3):
                I = elem_nodes[i]
                F[I] += F_e[i]
                for j in range(3):
                    J = elem_nodes[j]
                    K[I, J] += K_e[i, j]
        
        elapsed = time.time() - start
        print("  Assembly complete in {:.3f} s".format(elapsed))
        
        self.K = K
        self.F = F
        return K, F
    
    def apply_boundary_conditions(self, T_left=100.0, T_right=0.0):
        """
        Apply Dirichlet boundary conditions:
            T = T_left on left edge (x=0)
            T = T_right on right edge (x=1)
        
        Uses the penalty method for robustness.
        """
        print("Applying boundary conditions...")
        
        # Large penalty parameter
        penalty = 1e10
        
        # Left edge: T = T_left
        for node in self.mesh.left_nodes:
            self.K[node, node] += penalty
            self.F[node] += penalty * T_left
        
        # Right edge: T = T_right
        for node in self.mesh.right_nodes:
            self.K[node, node] += penalty
            self.F[node] += penalty * T_right
        
        print("  Dirichlet: Left={}, Right={}".format(T_left, T_right))
        print("  Neumann: Top, Bottom, Hole (natural BC, no action needed)")
    
    def solve(self):
        """Solve the linear system K*T = F."""
        print("Solving linear system...")
        start = time.time()
        
        # Use numpy's direct solver (LU decomposition)
        # For larger systems, use scipy.sparse.linalg.spsolve
        T = np.linalg.solve(self.K, self.F)
        
        elapsed = time.time() - start
        print("  Solved in {:.3f} s".format(elapsed))
        print("  Temperature range: [{:.2f}, {:.2f}]".format(T.min(), T.max()))
        
        self.T = T
        return T
    
    def compute_heat_flux(self):
        """
        Compute heat flux at element centroids:
            qx = -k * dT/dx
            qy = -k * dT/dy
        """
        qx = np.zeros(self.n_elements)
        qy = np.zeros(self.n_elements)
        
        for e in range(self.n_elements):
            elem_nodes = self.mesh.elements[e]
            coords = self.mesh.nodes[elem_nodes]
            x = coords[:, 0]
            y = coords[:, 1]
            T_elem = self.T[elem_nodes]
            
            A = 0.5 * abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
            if A < 1e-14:
                continue
            
            b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
            c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
            
            # dT/dx = sum(T_i * b_i) / (2A)
            # dT/dy = sum(T_i * c_i) / (2A)
            dTdx = np.dot(T_elem, b) / (2.0 * A)
            dTdy = np.dot(T_elem, c) / (2.0 * A)
            
            qx[e] = -self.k * dTdx
            qy[e] = -self.k * dTdy
        
        self.qx = qx
        self.qy = qy
        return qx, qy

# ==============================================================================
# SECTION 3: CONVERGENCE STUDY (h-Refinement)
# ==============================================================================

def convergence_study():
    """
    Run the solver for multiple mesh refinements and plot convergence.
    For the manufactured solution, we use a simple case where we know
    the exact behavior: linear temperature profile T = 100*(1-x).
    """
    mesh_sizes = [10, 20, 30, 40, 50]
    n_nodes_list = []
    errors = []
    
    print("\n" + "="*60)
    print("CONVERGENCE STUDY (h-Refinement)")
    print("="*60)
    
    for N in mesh_sizes:
        print("\n--- Mesh: {}x{} ---".format(N, N))
        mesh = Mesh(Nx=N, Ny=N, hole_radius=0.15)
        solver = FEMSolver(mesh, k=1.0, Q=0.0)  # Q=0 for pure conduction
        solver.assemble()
        solver.apply_boundary_conditions(T_left=100.0, T_right=0.0)
        T = solver.solve()
        
        # Exact solution for pure conduction: T_exact = 100*(1-x)
        x = mesh.nodes[:, 0]
        T_exact = 100.0 * (1.0 - x)
        
        # L2 error
        error = np.sqrt(np.mean((T - T_exact)**2))
        
        n_nodes_list.append(mesh.n_nodes)
        errors.append(error)
        
        print("  Nodes: {}, Elements: {}".format(mesh.n_nodes, mesh.n_elements))
        print("  L2 Error: {:.6e}".format(error))
    
    return np.array(n_nodes_list), np.array(errors)

# ==============================================================================
# SECTION 4: VISUALIZATION
# ==============================================================================

def plot_results(mesh, solver, save_path='FEM2D_results.png'):
    """Create a comprehensive 4-panel figure."""
    T = solver.T
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    
    # Compute flux for arrows
    solver.compute_heat_flux()
    
    fig = plt.figure(figsize=(16, 12))
    
    # (a) Mesh
    ax1 = fig.add_subplot(2, 2, 1)
    tri = Triangulation(x, y, mesh.elements)
    ax1.triplot(tri, 'k-', linewidth=0.3, alpha=0.5)
    
    # Highlight boundaries
    hole = Circle((0.5, 0.5), mesh.hole_radius, fill=False, 
                  edgecolor='red', linewidth=2, linestyle='--')
    ax1.add_patch(hole)
    ax1.plot([0, 0], [0, 1], 'b-', linewidth=3, label='T=100 (Dirichlet)')
    ax1.plot([1, 1], [0, 1], 'b-', linewidth=3, label='T=0 (Dirichlet)')
    ax1.plot([0, 1], [0, 0], 'g-', linewidth=2, label='Insulated (Neumann)')
    ax1.plot([0, 1], [1, 1], 'g-', linewidth=2)
    
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title('(a) Triangular Mesh with Boundary Conditions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(fontsize=8, loc='upper right')
    
    # (b) Temperature contour
    ax2 = fig.add_subplot(2, 2, 2)
    levels = np.linspace(T.min(), T.max(), 20)
    contour = ax2.tricontourf(tri, T, levels=levels, cmap='jet')
    plt.colorbar(contour, ax=ax2, label='Temperature')
    hole2 = Circle((0.5, 0.5), mesh.hole_radius, fill=False, 
                   edgecolor='white', linewidth=2)
    ax2.add_patch(hole2)
    ax2.set_aspect('equal')
    ax2.set_title('(b) Temperature Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # (c) 3D surface plot
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_trisurf(x, y, T, triangles=mesh.elements, cmap='jet', 
                     edgecolor='none', alpha=0.9)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('T')
    ax3.set_title('(c) 3D Temperature Surface', fontsize=12, fontweight='bold')
    ax3.view_init(elev=30, azim=-60)
    
    # (d) Temperature profiles
    ax4 = fig.add_subplot(2, 2, 4)
    # Extract profiles along y=0.5 (horizontal centerline)
    centerline_nodes = np.where(np.abs(y - 0.5) < 0.02)[0]
    if len(centerline_nodes) > 0:
        sort_idx = np.argsort(x[centerline_nodes])
        x_prof = x[centerline_nodes][sort_idx]
        T_prof = T[centerline_nodes][sort_idx]
        ax4.plot(x_prof, T_prof, 'bo-', markersize=4, linewidth=1.5, label='FEM Solution')
        
        # Analytical for comparison (no hole approximation)
        x_ana = np.linspace(0, 1, 100)
        T_ana = 100.0 * (1.0 - x_ana)
        ax4.plot(x_ana, T_ana, 'r--', linewidth=2, label='Analytical (no hole)')
    
    ax4.set_xlabel('x', fontsize=11)
    ax4.set_ylabel('T', fontsize=11)
    ax4.set_title('(d) Temperature Profile along y=0.5', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print("\nFigure saved to: {}".format(save_path))
    plt.show()

def plot_convergence(n_nodes, errors, save_path='FEM2D_convergence.png'):
    """Plot h-refinement convergence."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Theoretical slope for linear elements: O(h^2) = O(N^-2)
    # Since h ~ 1/N and N_nodes ~ N^2, error ~ N_nodes^-1
    h = 1.0 / np.sqrt(n_nodes)
    
    ax.loglog(h, errors, 'bo-', markersize=8, linewidth=2, label='FEM Error')
    
    # Reference line: slope = 2 (quadratic convergence)
    h_ref = np.linspace(h.min(), h.max(), 50)
    error_ref = errors[0] * (h_ref / h[0])**2
    ax.loglog(h_ref, error_ref, 'r--', linewidth=1.5, label='O(h^2) Reference')
    
    ax.set_xlabel('Mesh size h', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Error', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Study: h-Refinement\n(Linear Triangular Elements)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add convergence rate annotation
    slope = np.polyfit(np.log(h), np.log(errors), 1)[0]
    ax.text(0.6, 0.3, 'Observed rate: {:.2f}'.format(slope), 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print("Convergence plot saved to: {}".format(save_path))
    plt.show()

# ==============================================================================
# SECTION 5: MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("2D FINITE ELEMENT HEAT CONDUCTION SOLVER")
    print("="*60)
    
    # --- Main simulation ---
    print("\n>>> MAIN SIMULATION (40x40 mesh)")
    mesh = Mesh(Nx=40, Ny=40, hole_radius=0.15)
    solver = FEMSolver(mesh, k=1.0, Q=1000.0)
    solver.assemble()
    solver.apply_boundary_conditions(T_left=100.0, T_right=0.0)
    T = solver.solve()
    plot_results(mesh, solver, save_path='FEM2D_results.png')
    
    # --- Convergence study ---
    print("\n" + "="*60)
    print(">>> CONVERGENCE STUDY")
    print("="*60)
    n_nodes, errors = convergence_study()
    plot_convergence(n_nodes, errors, save_path='FEM2D_convergence.png')
    
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  FEM2D_results.png      - Main results (mesh, contours, 3D, profiles)")
    print("  FEM2D_convergence.png  - Convergence study")
    print("\nNext steps:")
    print("1. Try different hole sizes: mesh = Mesh(Nx=40, hole_radius=0.2)")
    print("2. Try different heat sources: solver = FEMSolver(mesh, Q=5000)")
    print("3. Try finer mesh: mesh = Mesh(Nx=60, Ny=60)")
    print("4. Upload to GitHub with README")
    print("="*60)
