use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

#[derive(Clone)]
pub struct TriangleMesh {
    pub vertices: Array2<f64>,      // (N, 2)
    pub triangles: Array2<usize>,   // (M, 3)

    // Mesh info properties
    pub areas: Array1<f64>,         // (M)
    pub vertex_neighbor_tris: Vec<Vec<usize>>, // (N) 
}

impl TriangleMesh {
    pub fn new_beam(width : f64, height : f64, shape : (usize, usize)) -> TriangleMesh {
        let (vertices, triangles) = Self::make_beam_mesh(width, height, shape);
         
        let areas = Self::compute_triangle_areas(&vertices, &triangles);
        
        // verify that all areas are positive
        if areas.iter().any(|&a| a <= 0.0) {
            panic!("Could not create mesh! Triangle with negative area found");
        }
        
        let vertex_neighbor_tris = Self::compute_vertex_triangle_adjacency(&vertices, &triangles);

        TriangleMesh {vertices, triangles, areas, vertex_neighbor_tris}
    }
    
    pub fn new_ball(res: usize) -> TriangleMesh {
        let (vertices, triangles) = Self::make_circle_mesh(res);

        let areas = Self::compute_triangle_areas(&vertices, &triangles);
        
        // verify that all areas are positive
        if areas.iter().any(|&a| a <= 0.0) {
            panic!("Could not create mesh! Triangle with negative area found");
        }
        
        let vertex_neighbor_tris = Self::compute_vertex_triangle_adjacency(&vertices, &triangles);

        TriangleMesh {vertices, triangles, areas, vertex_neighbor_tris}
    }

    fn make_circle_mesh(res: usize) -> (Array2<f64>, Array2<usize>) {
        // idea: https://stackoverflow.com/questions/53406534/procedural-circle-mesh-with-uniform-faces 
        let mut vertices = Vec::<Array1<f64>>::new();
        let mut triangles = Vec::<Array1<usize>>::new();
        
        let d: f64 = 1.0 / (res as f64);
        // build vertices
        vertices.push(array![0.0, 0.0]); // start with center vertex at (0,0)
        
        for circle in 0..res {
            let angle_step = (PI * 2.0) / ((circle as f64 + 1.0) * 6.0);
            for point in 0..((circle+1)*6) {
                vertices.push(array![
                    (angle_step * (point as f64)).cos() * d * ((circle as f64) + 1.0),
                    (angle_step * (point as f64)).sin() * d * ((circle as f64) + 1.0)]);
            }
        }
        
        // convert vertices to Array2
        let vertices = Array2::from_shape_vec((vertices.len(), 2),
        vertices.into_iter().flat_map(|row| row.into_iter()).collect())
            .unwrap();

        // build triangles
        for circle in 0..res {
            let c = circle as isize;
            let mut other = 0;
            for point in 0..((circle+1)*6) {
                if point % (circle+1) != 0 {
                    // Create 2 triangles
                    let tri = vec![
                        Self::get_point_index(c-1, other+1),
                        Self::get_point_index(c-1, other),
                        Self::get_point_index(c, point)
                    ];
                    Self::make_ccw(&vertices, tri.clone()); 
                    triangles.push(array![tri[0], tri[1], tri[2]]);
                    let tri = vec![
                        Self::get_point_index(c, point),
                        Self::get_point_index(c, point+1),
                        Self::get_point_index(c-1, other+1)
                    ];
                    Self::make_ccw(&vertices, tri.clone()); 
                    triangles.push(array![tri[0], tri[1], tri[2]]); 
                    other += 1;
                } else {
                    // Create 1 inverse triangle
                    let tri = vec![
                        Self::get_point_index(c, point),
                        Self::get_point_index(c, point+1),
                        Self::get_point_index(c-1, other)
                    ];
                    let tri = Self::make_ccw(&vertices, tri.clone());
                    triangles.push(array![tri[0], tri[1], tri[2]]); 
                }
            } 
        }
        // convert triangles to Array2
        let triangles = Array2::from_shape_vec((triangles.len(), 3),
        triangles.into_iter().flat_map(|row| row.into_iter()).collect())
            .unwrap();

        (vertices, triangles)
    }
    
    fn signed_area(
        vertices: &Array2<f64>,
        i0: usize,
        i1: usize,
        i2: usize,
    ) -> f64 { 
        let xi = vertices[[i0, 0]];
        let yi = vertices[[i0, 1]];
        let xj = vertices[[i1, 0]];
        let yj = vertices[[i1, 1]];
        let xk = vertices[[i2, 0]];
        let yk = vertices[[i2, 1]];

        // ½[(x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)]
        ((xj - xi)*(yk - yi) - (xk - xi)*(yj - yi)) * 0.5

    }
    fn make_ccw(
        vertices: &Array2<f64>,
        tri: Vec<usize>,
    ) -> Vec<usize> {
        if Self::signed_area(vertices, tri[0], tri[1], tri[2]) <= 0.0 {
            println!("new tri: {:?}", tri);
            return vec![tri[0], tri[2], tri[1]]
        } else {
            return tri
        }

    }

    fn make_beam_mesh(width : f64, height : f64, shape : (usize, usize)) -> (Array2<f64>, Array2<usize>) {
        if width < 0.0 || height < 0.0 { panic!("Could not create mesh! Width/height cannot be negative") }
        let x0 = -width/2.0;
        let y0 = -height/2.0;
        let i_max = shape.0;
        let j_max = shape.1;
        let dx = width/(i_max as f64);
        let dy = height/(j_max as f64);
        let vert_count = (i_max+1) * (j_max+1);

        let mut v = Array2::<f64>::zeros((vert_count, 2));
        let mut t = Array2::<usize>::zeros((2*i_max*j_max, 3));
        
        // build vertices
        for j in 0..(j_max+1) {
            for i in 0..(i_max+1) {
                let k: usize = i + j*(i_max + 1);
                v[[k,0]] = x0 + (i as f64)*dx;
                v[[k,1]] = y0 + (j as f64)*dy;   
            }
        }
        // build triangles
        for j in 0..j_max {
            for i in 0..i_max {
                let k00: usize = i + j*(i_max+1); 
                let k01: usize = (i+1) + j*(i_max+1);
                let k10: usize = i + (j+1)*(i_max+1);
                let k11: usize = (i+1) + (j+1)*(i_max+1);
                let e: usize = 2 * (i+j*i_max);
                if (i + j + 1)%2 != 0 {
                    t.slice_mut(s![e, ..]).assign(&array![k00, k01, k11]);
                    t.slice_mut(s![e+1, ..]).assign(&array![k00, k11, k10]);
                } else {
                    t.slice_mut(s![e, ..]).assign(&array![k10, k00, k01]);
                    t.slice_mut(s![e+1, ..]).assign(&array![k10, k01, k11]);
                }
            }
        }
        
        (v, t)
    }
    
    fn compute_triangle_areas(vertices: &Array2<f64>, triangles: &Array2<usize>) -> Array1<f64> {
        triangles
            .outer_iter()
            .map(|tri| {
                Self::signed_area(vertices, tri[0], tri[1], tri[2])
            })
            .collect()
    }
    fn compute_vertex_triangle_adjacency(vertices: &Array2<f64>, triangles: &Array2<usize>) -> Vec<Vec<usize>> {
        let vertex_count: usize = vertices.shape()[0];
        let mut vertex_neighbor_tris: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];

        for (triangle_idx, tri) in triangles.outer_iter().enumerate() {
            for &v in tri.iter() { 
                vertex_neighbor_tris[v].push(triangle_idx);
            }
        }
        vertex_neighbor_tris
    }
    
    fn get_point_index(circle: isize, point: usize) -> usize {
        if circle < 0 { return 0; }
        //println!("circle {}", circle);
        let c = circle as usize;
        let x = point % ((c + 1) * 6);
        return 3 * c * (c + 1) + x + 1;
    }

}
