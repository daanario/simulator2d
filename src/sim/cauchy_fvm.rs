use ndarray::prelude::*;
use crate::mesh::*;
use crate::cv::*;
use ndarray_linalg::Inverse;
use ndarray::stack;
use ndarray_linalg::Trace;
use std::time::Instant;
use rayon::prelude::*;
use std::collections::HashMap;

use plotters::prelude::*;
use plotters::prelude::full_palette::PINK; 
//use plotters_gtk4::Paintable;
//use plotters_gtk4::PaintableBackend;

use lazy_static::lazy_static;
use plotters_arrows::ThinArrow;

#[derive(Clone)]
struct Material {
    young_modulus: f64, // Young's modulus, 
    nu: f64,            // Poisson ratio
    rho: f64,           // material density
}

pub struct CauchyFVM {
    // holds data for a simulator based on the FVM applied to Cauchy's equation
    num_nodes: usize,
    pub sim_mesh: TriangleMesh,
    material_coords: Array2<f64>,
    control_volumes: Vec<MedianCentroidControlVolume>, 
    pub t: f64, // current time
    dt: f64, // delta time

    // material parameters  
    material: Material,
    lambda: f64,        // First Lamé coefficient
    mu: f64,            // Second Lamé coefficient
    
    // precomputed D_0 matrix inverses for all nodes and their elements
    inv_d0: Vec<Vec<Array2<f64>>>, 
    
    // force vector for the traction surface
    traction_force_vector: Array1<f64>,
    traction_boundary: Vec<usize>,

    immovable_boundary: Vec<usize>,

    // Holds current velocity for each node
    velocities: Array2<f64>,
}

impl CauchyFVM {
    pub fn new(mesh: &TriangleMesh, material_name: &str, dt: f64) -> CauchyFVM {
        let num_nodes = mesh.vertices.nrows();
        let sim_mesh = mesh.clone();
        let material_coords = mesh.clone().vertices; // material coordinates

        let mut control_volumes = Vec::<MedianCentroidControlVolume>::new();
        
        // generate control volumes for all nodes
        for node_idx in 0..num_nodes {
            control_volumes.push(MedianCentroidControlVolume::new(node_idx, &sim_mesh));
        }
        
        // set time and delta time
        let t = 0.0;
        let dt = dt;
        
        // set material parameters
        let material = MATERIALS.get(material_name).unwrap().clone();

        let lambda = (material.young_modulus * material.nu) / ((1.0+material.nu)*(1.0-2.0*material.nu));
        let mu = material.young_modulus / (2.0 * (1.0+material.nu));
        
        // precompute (D_0)^{-1} 
        let inv_d0 = Self::precompute_d0_invs(num_nodes, &sim_mesh, &control_volumes);
        
        let traction_force_vector = array![0.0, -10e4];
        
        // set of nodes on the traction boundary
        let traction_boundary: Vec<_> = (0..material_coords.nrows())
            .filter_map(|node_idx| {
                if material_coords[[node_idx, 0]] > 2.99 {
                    Some(node_idx)
                } else {
                    None
                }
            })
            .collect();
        
        // set of nodes on the immovable boundary
        let immovable_boundary: Vec<_> = (0..material_coords.nrows())
            .filter_map(|node_idx| {
                if material_coords[[node_idx, 0]] < -2.99 {
                    Some(node_idx)
                } else {
                    None
                }
            })
            .collect();

        // initial velocity is (0,0) for all nodes
        let velocities = Array2::<f64>::zeros((num_nodes, 2));

        CauchyFVM {
            num_nodes,
            sim_mesh,
            material_coords,
            control_volumes,
            dt,
            t, 
            material,
            lambda,
            mu,
            inv_d0,
            traction_force_vector,
            traction_boundary,
            immovable_boundary,
            velocities,
        }
    }
    
    pub fn compute_stress_tensors(&self) -> Vec<Vec<Array2<f64>>> {
        // compute stress tensors in parallel across nodes
        (0..self.num_nodes).into_par_iter()
            .map(|node_idx| {
                let cv = &self.control_volumes[node_idx];
                cv.neighbor_tri_ids.iter().enumerate()
                    .map(|(local_tri_idx, &tri_id)| {
                        let triangle = &self.sim_mesh.triangles.row(tri_id);

                        let i = triangle[0];
                        let j = triangle[1];
                        let k = triangle[2];

                        let pi = self.sim_mesh.vertices.row(i);
                        let pj = self.sim_mesh.vertices.row(j);
                        let pk = self.sim_mesh.vertices.row(k);

                        let gij = &pj - &pi;
                        let gik = &pk - &pi;
                        let d0_elem = stack![Axis(1), gij, gik];

                        let inv_d0_elem = &self.inv_d0[node_idx][local_tri_idx];

                        let fe = d0_elem.dot(inv_d0_elem);
                        let ee: Array2<f64> = 0.5 * (&fe.t().dot(&fe) - Array::eye(2));
                        // compute second Piola-Kirchoff stress tensor
                        let se: Array2<f64> = (self.lambda * ee.trace().expect("LinAlg Error! Could not find trace")) * Array::eye(2) + (2.0 * self.mu * ee);
                        // compute first Piola-Kirchoff stress tensor
                        let pe: Array2<f64> = fe.dot(&se);

                        pe
                    })
                .collect::<Vec<_>>()
            })
        .collect()
    }

    fn compute_elastic_forces(&self) -> Array2<f64> {
        let num_nodes = self.num_nodes;
        let stress_tensors = Self::compute_stress_tensors(self);
        
        // compute elastic forces in parallel across nodes
        let elastic_forces: Vec<Array1<f64>> = (0..num_nodes).into_par_iter()
            .map(|node_idx| {
                let cv = &self.control_volumes[node_idx];
                let lij = &cv.lij;
                let lik = &cv.lik;
                let nij = &cv.nij;
                let nik = &cv.nik;

                let mut elastic_force = Array1::<f64>::zeros(2);

                for (local_elem_idx, _) in cv.neighbor_tri_ids.iter().enumerate() {
                    let pe = &stress_tensors[node_idx][local_elem_idx];
                    let f_elem = -0.5 * pe.dot(&nij.slice(s![local_elem_idx, ..])) * lij[local_elem_idx]
                        - 0.5 * pe.dot(&nik.slice(s![local_elem_idx, ..])) * lik[local_elem_idx];
                    elastic_force += &f_elem;
                }
                elastic_force
            })
        .collect();

        // Convert back to Array2
        let elastic_forces = Array2::from_shape_vec((num_nodes, 2),
        elastic_forces.into_iter().flat_map(|row| row.into_iter()).collect())
            .unwrap();
        
        elastic_forces
    }
    
    fn compute_total_forces (&self) -> Array2<f64> { 
        let elastic_forces = Self::compute_elastic_forces(self);

        // compute total forces in parallel across nodes
        let total_forces: Vec<Array1<f64>> = (0..self.num_nodes).into_par_iter()
            .map(|node_idx| {
                let cv = &self.control_volumes[node_idx];

                let mut traction_force = Array1::<f64>::zeros(2); 
                let mut penalty_force = Array1::<f64>::zeros(2); 
                if self.traction_boundary.contains(&node_idx) {
                    let force_array = &self.traction_force_vector * cv.area;
                    traction_force.assign(&force_array);  
                }
                /*
                if self.sim_mesh.vertices[[node_idx, 1]] < -500.5 {
                    let node_pos = array![0.0, self.sim_mesh.vertices[[node_idx, 1]]];
                    let penetration = -1.0 * node_pos;
                    penalty_force = 1e7 * penetration * cv.area; // stiffness * penetration
                }
                */
                let gravity = array![0.0, -9.8e2] * cv.area;
                &elastic_forces.row(node_idx) + &traction_force + &gravity// + &penalty_force
            })
            .collect();
        
        // Convert back to Array2
        let total_forces = Array2::from_shape_vec((self.num_nodes, 2),
        total_forces.into_iter().flat_map(|row| row.into_iter()).collect())
            .unwrap();    
        total_forces
    }

    fn compute_velocities(&self) -> Array2<f64> {     
        let forces = Self::compute_total_forces(&self);

        // compute nodal velocities in parallel across nodes
        let velocities: Vec<Array1<f64>> = (0..self.num_nodes).into_par_iter()
            .map(|node_idx| {
                let cv = &self.control_volumes[node_idx];
                let nodal_mass = self.material.rho * cv.area;
                let old_velocity = self.velocities.row(node_idx);
                if self.immovable_boundary.contains(&node_idx) {
                    array![0.0, 0.0]
                } else {
                    &old_velocity + (self.dt / nodal_mass) * &forces.row(node_idx)
                } 
            })
            .collect();

        // Convert back to Array2
        let velocities = Array2::from_shape_vec((self.num_nodes, 2),
        velocities.into_iter().flat_map(|row| row.into_iter()).collect())
            .unwrap();    
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
    
    pub fn set_material(&mut self, name: &str) -> () {
        let material = MATERIALS.get(name).unwrap().clone();
        let lambda = (material.young_modulus * material.nu) / ((1.0+material.nu)*(1.0-2.0*material.nu));
        let mu = material.young_modulus / (2.0 * (1.0+material.nu));

        self.material = material;
        self.lambda = lambda;
        self.mu = mu;
    }

    pub fn set_traction_force(&mut self, force_vector: Array1<f64>) -> () {
        self.traction_force_vector = force_vector;
    }
    
    pub fn set_traction_boundary(&mut self, boundary_name: &str) -> () {
        match boundary_name {
            "right" => self.traction_boundary = (0..self.material_coords.nrows())
                .filter_map(|node_idx| {
                    if self.material_coords[[node_idx, 0]] > 2.99 {
                        Some(node_idx)
                    } else {
                        None
                    }
                })
                .collect(),
            "down" => self.traction_boundary = (0..self.material_coords.nrows())
                .filter_map(|node_idx| {
                    if self.material_coords[[node_idx, 1]] < -0.99 {
                        Some(node_idx)
                    } else {
                        None
                    }
                })
                .collect(),
            "up" => self.traction_boundary = (0..self.material_coords.nrows())
                    .filter_map(|node_idx| {
                        if self.material_coords[[node_idx, 1]] > 0.99 {
                            Some(node_idx)
                        } else {
                            None
                        }
                    })
                .collect(),
            _ => ()
        }
    }
    
    pub fn set_immovable_boundary(&mut self, boundary_name: &str) -> () {
        match boundary_name {
            "left" => self.immovable_boundary = (0..self.material_coords.nrows())
                .filter_map(|node_idx| {
                    if self.material_coords[[node_idx, 0]] < -2.99 {
                        Some(node_idx)
                    } else {
                        None
                    }
                })
                .collect(),
            "leftright" => self.immovable_boundary = (0..self.material_coords.nrows())
                .filter_map(|node_idx| {
                    if self.material_coords[[node_idx, 0]] < -2.99
                        || self.material_coords[[node_idx, 0]] > 2.99 {
                            Some(node_idx)
                        } else {
                            None
                        }
                })
                .collect(),
            _ => println!("Error: this boundary does not exist! \n
                Immovable boundaries: {{ 'left', 'leftright' }}")

        }
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
/*
    pub fn display_on_paintable(&self, paintable: &Paintable) -> () {

        let backend = PaintableBackend::new(paintable);
        let root = backend.into_drawing_area();
        root.fill(&BLACK).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Beam mesh, t={time:.*}s", 3, time=self.t), ("sans-serif", 12, &WHITE))
            .build_cartesian_2d(-40.0..60.0, -40.0..40.0)
            .unwrap();
         
        let mesh = &self.sim_mesh;
        let material_coords = &self.material_coords;

        for tri in mesh.triangles.outer_iter() {
            let triangle = vec![
                (mesh.vertices[[tri[0], 0]], mesh.vertices[[tri[0], 1]]),
                (mesh.vertices[[tri[1], 0]], mesh.vertices[[tri[1], 1]]),
                (mesh.vertices[[tri[2], 0]], mesh.vertices[[tri[2], 1]]),
                (mesh.vertices[[tri[0], 0]], mesh.vertices[[tri[0], 1]]), // close loop
            ];

            chart
                .draw_series(std::iter::once(PathElement::new(triangle, &WHITE)))
                .unwrap();
            }
        
        // draw immovable boundary surface as a different color
        let mut boundary_left_vec = Vec::<(f64, f64)>::new();
        for node_idx in 0..mesh.vertices.nrows() {
            if self.immovable_boundary.contains(&node_idx) && material_coords[[node_idx, 0]] < -2.99 {
                boundary_left_vec.push((mesh.vertices[[node_idx, 0]], mesh.vertices[[node_idx, 1]]));
            }
        }
        
        let mut boundary_right_vec = Vec::<(f64, f64)>::new();
        for node_idx in 0..mesh.vertices.nrows() {
            if self.immovable_boundary.contains(&node_idx) && material_coords[[node_idx, 0]] > 2.99 {
                boundary_right_vec.push((mesh.vertices[[node_idx, 0]], mesh.vertices[[node_idx, 1]]));
            }
        }

        chart 
            .draw_series(std::iter::once(PathElement::new(boundary_left_vec, &CYAN)))
            .unwrap();  
        chart 
            .draw_series(std::iter::once(PathElement::new(boundary_right_vec, &CYAN)))
            .unwrap();  

        // draw traction surface as a different color
        let mut traction_vec = Vec::<(f64, f64)>::new();
        for node_idx in 0..mesh.vertices.nrows() {
            if self.traction_boundary.contains(&node_idx) {
                traction_vec.push((mesh.vertices[[node_idx, 0]], mesh.vertices[[node_idx, 1]]));
            }
        }

        chart 
            .draw_series(std::iter::once(PathElement::new(traction_vec, &RED)))
            .unwrap();  
        
        // draw velocity vectors       
        chart 
            .draw_series((0..self.num_nodes)
                .into_iter()
                .map(|node_idx| {
                    let arrow_size = 0.15;
                    let x = mesh.vertices[[node_idx, 0]];
                    let y = mesh.vertices[[node_idx, 1]];
                    let dx = arrow_size * self.velocities[[node_idx, 0]];
                    let dy = arrow_size * self.velocities[[node_idx, 1]];
                    ThinArrow::new((x, y), (x + dx, y + dy), &GREEN)
                }))
                .unwrap();
        
        // draw force vectors        
        chart 
            .draw_series(self.traction_boundary
                .clone()
                .into_iter()
                .map(|node_idx| {
                    let arrow_size = 1e-5;
                    let x = mesh.vertices[[node_idx, 0]];
                    let y = mesh.vertices[[node_idx, 1]];
                    let dx = arrow_size * self.traction_force_vector[0];
                    let dy = arrow_size * self.traction_force_vector[1];
                    ThinArrow::new((x, y), (x + dx, y + dy), &RED)
                }))
                .unwrap();
        
        // draw floor
        let x_values = vec![-6.0, 6.0];
        chart.draw_series(LineSeries::new(x_values.into_iter().map(|x| (x, -3.5)), &WHITE)).unwrap();

        root.present().unwrap();
    }
*/
}

lazy_static! {
    static ref MATERIALS: HashMap<&'static str, Material> = HashMap::from([
            ("default", Material {young_modulus: 10e5, nu: 0.3, rho: 1000.0}),
            ("rubber", Material {young_modulus: 0.01e9, nu: 0.48, rho: 1050.0}),
        ]);
}

