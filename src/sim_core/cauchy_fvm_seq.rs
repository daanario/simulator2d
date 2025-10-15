// Sequential (Slow! version of the CauchyFVM simulator)

use ndarray::prelude::*;
use crate::mesh::*;
use crate::cv::*;
use ndarray_linalg::Inverse;
use ndarray::stack;
use ndarray_linalg::Trace;
use std::time::Instant;
use rayon::prelude::*;

pub struct CauchyFVM {
    // holds data for a simulator based on the FVM applied to Cauchy's equation
    num_nodes: usize,
    pub sim_mesh: TriangleMesh,
    control_volumes: Vec<MedianCentroidControlVolume>, 
    t: f64, // current time
    dt: f64, // delta time
    
    // material parameters
    young_modulus: f64, // Young's modulus, E
    nu: f64,            // Poisson ratio
    rho: f64,           // material density
    lambda: f64,        // First Lamé coefficient
    mu: f64,            // Second Lamé coefficient
    
    // precomputed D_0 matrix inverses for all nodes and their elements
    inv_d0: Vec<Vec<Array2<f64>>>, 
    
    // Holds current velocity for each node
    velocities: Array2<f64>,
}

impl CauchyFVM {
    pub fn new(mesh: &TriangleMesh) -> CauchyFVM {
        let num_nodes = mesh.vertices.nrows();
        let sim_mesh = mesh.clone();
        
        let mut control_volumes = Vec::<MedianCentroidControlVolume>::new();
        
        // generate control volumes for all nodes
        for node_idx in 0..num_nodes {
            control_volumes.push(MedianCentroidControlVolume::new(node_idx, &sim_mesh));
        }
        
        // set time and delta time (hardcoded for now)
        let t = 0.0;
        let dt = 0.001;

        // set material parameters (hardcoded for now)
        //let young_modulus = 10e5;
        //let nu = 0.3;
        //let rho = 1000.0;
        let young_modulus = 0.01e9;
        let nu = 0.48;
        let rho = 1050.0;

        let lambda = (young_modulus * nu) / ((1.0+nu)*(1.0-2.0*nu));
        let mu = young_modulus / (2.0 * (1.0+nu));
        
        // precompute (D_0)^{-1} 
        let inv_d0 = Self::precompute_d0_invs(num_nodes, &sim_mesh, &control_volumes);

        // initial velocity is (0,0) for all nodes
        let velocities = Array2::<f64>::zeros((num_nodes, 2));

        CauchyFVM {
            num_nodes,
            sim_mesh,
            control_volumes,
            dt,
            t, 
            young_modulus,
            nu,
            rho,
            lambda,
            mu,
            inv_d0,
            velocities,
        }
    }
    
    pub fn compute_stress_tensors(&self) -> Vec<Vec<Array2<f64>>> {
        let num_nodes = self.num_nodes; 
        let mut stress_tensors = Vec::<Vec<Array2<f64>>>::new();

        for node_idx in 0..num_nodes {
            let cv = &self.control_volumes[node_idx]; 
            let mut stress_tensors_elements = Vec::<Array2<f64>>::new();

            for (local_tri_idx, tri_id) in cv.neighbor_tri_ids.clone().into_iter().enumerate() {
                let triangle = &self.sim_mesh.triangles.row(tri_id);
                
                let i = triangle[0];
                let j = triangle[1];
                let k = triangle[2];

                let pi = self.sim_mesh.vertices.slice(s![i, ..]); // vertex i 
                let pj = self.sim_mesh.vertices.slice(s![j, ..]); // vertex j 
                let pk = self.sim_mesh.vertices.slice(s![k, ..]); // vertex k
                
                let gij = &pj - &pi;
                let gik = &pk - &pi;

                let d0_elem = stack![Axis(1), gij, gik];

                let inv_d0_elem = &self.inv_d0[node_idx][local_tri_idx];

                let fe = d0_elem.dot(inv_d0_elem); // Matrix F^e (2 x 2)
                let ee: Array2<f64> = 0.5 * (&fe.t().dot(&fe) - Array::eye(2)); // Matrix E^e (2 x 2)
                let se = (self.lambda * &ee.trace().expect("LinAlg Error! Could not find trace")) * Array::eye(2) + (2.0 * self.mu * &ee); // Matrix S^e (2 x 2)

                let pe = fe.dot(&se); // Stress tensor for current triangle element: Matrix P^e (2 x 2)
                stress_tensors_elements.push(pe);
            }
            stress_tensors.push(stress_tensors_elements);
        }
        stress_tensors
    }

    fn compute_elastic_forces(&self) -> Array2<f64> {
        let num_nodes = self.num_nodes;
        let stress_tensors = Self::compute_stress_tensors(self);
        let mut elastic_forces = Array2::<f64>::zeros((num_nodes, 2));
        for node_idx in 0..num_nodes {
            let cv = &self.control_volumes[node_idx];
             
            let lij = &cv.lij; // Vector of size E
            let lik = &cv.lik; // Vector of size E
            let nij = &cv.nij; // (E x 2) matrix where E is the number of elements for node i
            let nik = &cv.nik; // (E x 2) matrix where E is the number of elements for node i

            for (local_elem_idx, _elem_id) in cv.neighbor_tri_ids.clone().into_iter().enumerate() {
                let pe = &stress_tensors[node_idx][local_elem_idx];
                let elastic_force_elem = -0.5 * pe.dot(&nij.slice(s![local_elem_idx, ..])) * lij[local_elem_idx]
                    - 0.5 * pe.dot(&nik.slice(s![local_elem_idx, ..])) * lik[local_elem_idx];
                let add_row = &elastic_forces.row(node_idx) + &elastic_force_elem;
                elastic_forces.row_mut(node_idx).assign(&add_row); // (N x 2) matrix, where N is number of nodes 
            }
        }
        elastic_forces
    }
    
    fn compute_total_forces (&self) -> Array2<f64> { 
        let mut total_forces = Array2::<f64>::zeros((self.num_nodes, 2));
        let elastic_forces = Self::compute_elastic_forces(self);
        
        for node_idx in 0..self.num_nodes {
            let cv = &self.control_volumes[node_idx];
            
            let mut traction_force = Array1::<f64>::zeros(2);
            if *&self.sim_mesh.vertices[[node_idx, 0]] > 2.99 { // hardcoded traction boundary
                let force_array = array![0.0, -10e4] * cv.area;
                traction_force.assign(&force_array);  
            }
            let add_row = &total_forces.row(node_idx) + traction_force + elastic_forces.row(node_idx);
            total_forces.row_mut(node_idx).assign(&add_row);
            
        }
        total_forces
    }

    fn compute_velocities(&self) -> Array2<f64> {
        let forces = Self::compute_total_forces(&self);
        let mut velocities = self.velocities.clone();
        for node_idx in 0..self.num_nodes {
            let cv = &self.control_volumes[node_idx];
            let nodal_mass = self.rho * cv.area;
            if *&self.sim_mesh.vertices[[node_idx, 0]] > -2.99 { // hardcoded immovable boundary
                let add_row = &velocities.row(node_idx) + ((self.dt/nodal_mass) * &forces.row(node_idx));
                velocities.row_mut(node_idx).assign(&add_row);
            } 
        }
        velocities
    }

    pub fn update(&mut self) -> () {
        // compute new velocities
        self.velocities = self.compute_velocities();
        
        // set new vertex positions
        for node_idx in 0..self.num_nodes {
            let pos_update = &self.sim_mesh.vertices.row(node_idx) + (&self.velocities.row(node_idx) * self.dt);
            self.sim_mesh.vertices.row_mut(node_idx).assign(&pos_update);
        }
        // step forward in time
        self.t += self.dt;
    }

    fn precompute_d0_invs(num_nodes: usize,
        sim_mesh: &TriangleMesh,
        control_volumes: &Vec<MedianCentroidControlVolume>) -> Vec<Vec<Array2<f64>>> {
        let mut inv_d0 = Vec::<Vec<Array2<f64>>>::new();
        
        for node_idx in 0..num_nodes { 
            let cv = &control_volumes[node_idx];
            let mut inv_d0_elements = Vec::<Array2<f64>>::new();
            
            for tri_id in cv.neighbor_tri_ids.clone() {
                let triangle = sim_mesh.triangles.row(tri_id);
                
                let i = triangle[0];
                let j = triangle[1];
                let k = triangle[2];
                
                let pi = sim_mesh.vertices.slice(s![i, ..]); // vertex i 
                let pj = sim_mesh.vertices.slice(s![j, ..]); // vertex j 
                let pk = sim_mesh.vertices.slice(s![k, ..]); // vertex k
                
                let gij = &pj - &pi;
                let gik = &pk - &pi;

                let d0_elem = stack![Axis(1), gij, gik]; // column stack gij and gik to
                                                                 // form D0
                inv_d0_elements.push(d0_elem.inv().expect("LinAlg Error! Matrix not invertible.")); // invert D0_e
                
            }
            inv_d0.push(inv_d0_elements); // append all D_0^{-1} for the elements of this node
        }
        inv_d0
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
