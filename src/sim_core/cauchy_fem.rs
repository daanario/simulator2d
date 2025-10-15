use ndarray::prelude::*;
use crate::mesh::*;
use crate::cv::*;
use ndarray_linalg::Inverse;
use ndarray::stack;
use ndarray_linalg::Trace;
use std::time::Instant;
use ndarray_linalg::Solve;
use std::collections::HashSet;
use ndarray::concatenate;

pub struct CauchyFEM {
    num_nodes: usize,
    num_elements: usize,
    pub sim_mesh: TriangleMesh,
    pub material_coords: Array2<f64>,
    // perhaps use an array holding each element?

    t: f64, // current time
    dt: f64, // delta time
    
    // material parameters
    young_modulus: f64, // Young's modulus, E
    nu: f64,            // Poisson ratio
    rho: f64,           // material density
    //lambda: f64,        // First Lamé coefficient
    //mu: f64,            // Second Lamé coefficient
    
    // precomputed E_0 matrix inverses for all elements
    inv_e0: Vec<Array2<f64>>, 
    
    // precomputed mass matrix
    mass: Array2<f64>,

    // Holds current velocity for each node
    velocities: Array2<f64>,
}

impl CauchyFEM {
    pub fn new(mesh: &TriangleMesh) -> CauchyFEM {
        // hardcoded material parameters 
        let young_modulus = 0.01e9;
        let nu = 0.48;
        let rho = 1050.0;
        
        let lambda = (young_modulus * nu) / ((1.0+nu)*(1.0-2.0*nu));
        let mu = young_modulus / (2.0 * (1.0+nu));
        
        let mut t = 0.0;
        let dt = 0.01;

        let mut sim_mesh = mesh.clone();
        let material_coords = sim_mesh.vertices.clone();
        
        let num_nodes = mesh.vertices.nrows();
        let num_elements = mesh.triangles.nrows();
        let inv_e0 = Self::precompute_e0_invs(num_elements, mesh);
        let mass = Self::precompute_mass(num_elements, mesh, rho);
        
        // initial velocity is (0,0) for all nodes
        let mut velocities = Array2::<f64>::zeros((num_nodes, 2));

        CauchyFEM {
            num_nodes,
            num_elements,
            sim_mesh,
            material_coords,
            t,
            dt,
            young_modulus,
            nu,
            rho,
            inv_e0,
            mass,
            velocities,
        }
    }
    
    pub fn update(&mut self) -> () {
        let new_vertices = self.compute_new_vertex_positions();
        self.sim_mesh.vertices = new_vertices;
        self.t += self.dt;
    }

    fn compute_ke(&self) -> Array2<f64> {
        // block array of 6 x 6 matrices for each element e
        let mut ke = Array2::<f64>::zeros((6, self.num_elements * 6));
        
        let nu = self.nu;
        // elasticity matrix
        let d = self.young_modulus/(1.0 - (nu * nu)) * array![[1.0, nu, 0.0],
                                                              [nu, 1.0, 0.0 ],
                                                              [0.0, 0.0, (1.0-nu)/2.0]];
        // TODO: parallelize over elements, perhaps also move corotational form in here
        for elem_idx in 0..self.num_elements {
            let triangle = self.sim_mesh.triangles.row(elem_idx);
            let i = triangle[0];
            let j = triangle[1];
            let k = triangle[2];

            let area = self.sim_mesh.areas[elem_idx];
            
            // get triangle coordinates
            let vertices = &self.sim_mesh.vertices;
            let xi = vertices[[i, 0]];
            let xj = vertices[[j, 0]];
            let xk = vertices[[k, 0]];
            let yi = vertices[[i, 1]];
            let yj = vertices[[j, 1]];
            let yk = vertices[[k, 1]];

            // compute spatial gradients of the barycentric coordinates
            let dw1dx = &yj - &yk;
            let dw1dy = &xk - &xj;
            let dw2dx = &yk - &yi;
            let dw2dy = &xi - &xk;
            let dw3dx = &yi - &yj;
            let dw3dy = &xj - &xi;

            let b = (1.0 / (2.0*area)) * array![[dw1dx, 0.0, dw2dx, 0.0, dw3dx, 0.0],
                           [0.0, dw1dy, 0.0, dw2dy, 0.0, dw3dy],
                           [dw1dy, dw1dx, dw2dy, dw2dx, dw3dy, dw3dx]];
           
            // Compute element stiffness matrix and store it in arrays of K^e's 
            let local_ke = b.t().dot(&d).dot(&b) * area;
            ke.slice_mut(s![.., 6*elem_idx..6*elem_idx+6]).assign(&local_ke);
        }
        ke
    }
    fn compute_corotational_form(&self, ke: Array2<f64>, inv_e0: Vec<Array2<f64>>) -> Array2<f64> {
        // TODO: finish the function
        let ke_prime = Array2::<f64>::zeros((6, self.num_elements * 6));
        
        for elem_idx in 0..self.num_elements {
            let triangle = self.sim_mesh.triangles.row(elem_idx);
            let vertices = &self.sim_mesh.vertices;

            let i = triangle[0];
            let j = triangle[1];
            let k = triangle[2];
            
            let pi = vertices.slice(s![i, ..]); // vertex i 
            let pj = vertices.slice(s![j, ..]); // vertex j 
            let pk = vertices.slice(s![k, ..]); // vertex k
            
        }
        ke_prime
    }
    
    fn compute_new_vertex_positions(&self) -> Array2<f64> {
        // TODO: Apply boundary conditions
        let num_nodes = self.num_nodes;

        // assemble global system of equations
        let dt = &self.dt;
        let mass = &self.mass;
        let ke = self.compute_ke();
        let k_matrix = self.matrix_assembly(ke);
        
        // current vertex positions
        let vertices = &self.sim_mesh.vertices;
        let vertices_x = vertices.column(0);
        let vertices_y = vertices.column(1);
        let flattened_vertices: Array1<f64> = concatenate![Axis(0), vertices_x, vertices_y]; 
        
        let material_coords_x = self.material_coords.column(0);
        let material_coords_y = self.material_coords.column(1);
        let flattened_material_coords: Array1<f64> = concatenate![Axis(0), material_coords_x, material_coords_y]; 

        // force vector of size (2N). First N entries are x, all following N+i entries are y.
        let mut f = Array1::<f64>::zeros(2 * self.num_nodes);

        // hardcoded body forces, just gravity for now (-9.8 in the y direction only)
        f.slice_mut(s![self.num_nodes..]).fill(-9.8);
        
        // add traction force only to the traction nodes
        let x_traction_vec: Vec<usize> = material_coords_x.indexed_iter()
            .filter(|&(_, &val)| val > 2.9)
            .map(|(idx, _)| idx)
            .collect();
        let x_traction = Array1::from_vec(x_traction_vec);
        let y_traction = &x_traction + num_nodes;
        let traction_arr = concatenate![Axis(0), x_traction, y_traction]; 
        let traction_indices = traction_arr.to_vec();
        
        // Fill in traction forces only at traction indices   
        for i in y_traction {
            if self.t < 1.50 {
                f[i] += -10e4;
            }
        } 

        // material forces
        let f0 = k_matrix.dot(&flattened_material_coords);
        
        // flatten velocities
        let velocities = &self.velocities;        
        let velocities_x = velocities.column(0);
        let velocities_y = velocities.column(1); 
        let flattened_velocities = concatenate![Axis(0), velocities_x, velocities_y]; 

        let mut a = mass + (dt * dt) * &k_matrix; 
        let mut b = mass.dot(&flattened_velocities) + (*dt) * (f - &k_matrix.dot(&flattened_vertices) + f0);
        
        // before solving, set boundary conditions
        
        // Collect indices where x < -2.9
        let x_indices_vec: Vec<usize> = material_coords_x.indexed_iter()
            .filter(|&(_, &val)| val < -2.9)
            .map(|(idx, _)| idx)
            .collect();
        let x_indices = Array1::from_vec(x_indices_vec);
        let y_indices = &x_indices + num_nodes;
        let indices_arr = concatenate![Axis(0), x_indices, y_indices]; 
        let indices = indices_arr.to_vec();
        
        let all_indices: HashSet<usize> = (0..2*num_nodes).collect();
        let constrained_indices: HashSet<usize> = indices.into_iter().collect();
        let free_indices: Vec<usize> = all_indices
            .difference(&constrained_indices)
            .copied()
            .collect(); 

        let af = a.select(Axis(0), &free_indices);
        let aff = af.select(Axis(1), &free_indices); 
        let bf = b.select(Axis(0), &free_indices);

        // solve the linear system
        let new_velocities = aff.solve_into(bf).unwrap();
        
        let mut full_new_velocities = Array1::<f64>::zeros(2*num_nodes);

        // Fill in values only at free indices
        for (i, &idx) in free_indices.iter().enumerate() {
            full_new_velocities[idx] = new_velocities[i];
        }

        // add update
        let vertices_update = &flattened_vertices + (*dt) * &full_new_velocities; 
        
        // split and column stack into 2D array and return
        let (xs, ys) = vertices_update.view().split_at(Axis(0), self.num_nodes);
        
        let x_view: ArrayView1<f64> = ArrayView1::from(xs);
        let y_view: ArrayView1<f64> = ArrayView1::from(ys);
        let mut new_vertices = Array2::<f64>::zeros((self.num_nodes, 2));

        new_vertices.slice_mut(s![.., 0]).assign(&x_view);
        new_vertices.slice_mut(s![.., 1]).assign(&y_view); 
        
        new_vertices
    }

    fn matrix_assembly(&self, ke: Array2<f64>) -> Array2<f64> {
        let num_nodes = self.sim_mesh.vertices.nrows();
        let mut k_matrix = Array2::<f64>::zeros((num_nodes*2, num_nodes*2));
        
        for elem_idx in 0..self.num_elements {
            let triangle = self.sim_mesh.triangles.row(elem_idx);
            let i = triangle[0];
            let j = triangle[1];
            let k = triangle[2];

            // Local order of vertex coordinates is i_x, i_y, j_x j_y, k_x, and  k_y. 
            // This is how local vertex indices (0,1,2,..,5) are mapped to global vertex
            // indices
            
            let gidx = array![i, num_nodes + i, j, num_nodes + j, k, num_nodes + k];
            
            for idx_i in 0..6 {
                for idx_j in 0..6 {
                    let global_i = gidx[idx_i];
                    let global_j = gidx[idx_j];

                    let local_ke_entry = &ke[[idx_i, idx_j + elem_idx*6]];
                    let k_matrix_update = &k_matrix[[gidx[idx_i], gidx[idx_j]]] + local_ke_entry;
                    k_matrix[[gidx[idx_i], gidx[idx_j]]] = k_matrix_update;
                }
            }
        }
        k_matrix
    }
    fn precompute_mass(num_elements: usize, mesh: &TriangleMesh, rho: f64) -> Array2<f64> {
        let areas = mesh.areas.clone();
        let num_nodes = mesh.vertices.nrows();

        // Store diagonal masses for each node
        let mut nodal_masses = vec![0.0; num_nodes];

        // Accumulate lumped mass per node
        for elem_idx in 0..num_elements {
            let triangle = mesh.triangles.row(elem_idx);
            let i = triangle[0];
            let j = triangle[1];
            let k = triangle[2];

            let m = (rho * areas[elem_idx]) / 3.0;

            nodal_masses[i] += m;
            nodal_masses[j] += m;
            nodal_masses[k] += m;
        }

        // Now build the full diagonal mass matrix of size (2N x 2N)
        let mut mass = Array2::<f64>::zeros((2*num_nodes, 2*num_nodes));

        for node_idx in 0..num_nodes {
            mass[[2*node_idx, 2*node_idx]] = nodal_masses[node_idx];
            mass[[2*node_idx+1, 2*node_idx+1]] = nodal_masses[node_idx];
        }

        mass
    }

    fn precompute_e0_invs(num_elements: usize,
        sim_mesh: &TriangleMesh) -> Vec<Array2<f64>> {
        let mut inv_e0_elements = Vec::<Array2<f64>>::new();
        
        for elem_idx in 0..num_elements {
            let triangle = sim_mesh.triangles.row(elem_idx);
            let i = triangle[0];
            let j = triangle[1];
            let k = triangle[2];
            
            let pi = sim_mesh.vertices.slice(s![i, ..]); // vertex i 
            let pj = sim_mesh.vertices.slice(s![j, ..]); // vertex j 
            let pk = sim_mesh.vertices.slice(s![k, ..]); // vertex k

            let gij = &pj - &pi;
            let gik = &pk - &pi;
            
            let e0_elem = stack![Axis(1), gij, gik]; // column stack gij and gik to form E_0^e for element e
                                                     
            inv_e0_elements.push(e0_elem.inv().expect("LinAlg Error! Matrix not invertible.")); // invert E_0^e
        }
        inv_e0_elements
    }

    pub fn benchmark(&mut self, iters: usize) -> () {
        // simple benchmarking function for checking how many updates per second we can get
        let now = Instant::now();
        let mut last_elapsed: u128 = 0;

        for i in 0..iters {
            self.update();

            if i % 100 == 0 {
                let elapsed_millisecs = now.elapsed().as_millis() - last_elapsed;
                last_elapsed += elapsed_millisecs;
                println!("updates/sec: {}", (1.0)/((elapsed_millisecs) as f64) * 100.0 * 1000.0);
            }
        }
    }
}
