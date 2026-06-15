use ndarray::prelude::*;
use ndarray::{Array1, Array2, concatenate};
use std::f64::consts::PI;

#[derive(Clone)]
pub struct TetrahedalMesh {
    pub vertices: Array2<f64>, // (N, 3)
    pub tetraheda: Array2<usize>,   // (M, 4)

    // Mesh info properties
    pub volumes: Array1<f64>,         // (M)
    pub vertex_neighbor_tetras: Vec<Vec<usize>>, // (N) 
}

impl TetrahedalMesh {
    pub fn new_beam(width : f64, height : f64, length : f64, shape : (usize, usize, usize)) -> TetrahedalMesh {
        let (vertices, tetraheda) = Self::make_beam_mesh(width, height, length, shape);
         
        let volumes = Self::compute_tetrahedon_volumes(&vertices, &tetraheda);
        
        // verify that all volumes are positive
        
        if volumes.iter().any(|&a| a <= 0.0) {
            panic!("Could not create mesh! Tetrahedon with negative volume found");
        }
        
        
        let vertex_neighbor_tetras = Self::compute_vertex_tetrahedon_adjacency(&vertices, &tetraheda);

        TetrahedalMesh {vertices, tetraheda, volumes, vertex_neighbor_tetras}
    }
    
    pub fn new_cube() -> TetrahedalMesh {
        let (vertices, tetraheda) = Self::make_cube_mesh();
        let volumes = Self::compute_tetrahedon_volumes(&vertices, &tetraheda);

        let vertex_neighbor_tetras = Self::compute_vertex_tetrahedon_adjacency(&vertices, &tetraheda);

        TetrahedalMesh { vertices, tetraheda, volumes, vertex_neighbor_tetras }

    }

    fn signed_volume(
        vertices: &Array2<f64>,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize
    ) -> f64 { 
        // vectors
        let a = vertices.row(i0).t().to_owned();
        let b = vertices.row(i1).t().to_owned();
        let c = vertices.row(i2).t().to_owned();
        let d = vertices.row(i3).t().to_owned();
        
        let ad = (&a - &d).insert_axis(Axis(1));
        let bd = (&b - &d).insert_axis(Axis(1));
        let cd = (&c - &d).insert_axis(Axis(1));

        let mat = concatenate![Axis(1), ad, bd, cd];
        
        // compute volume (panics if determinant fails!)
        (1.0/6.0) * ndarray_linalg::Determinant::det(&mat).unwrap() 

    }
    fn make_ccw(
        vertices: &Array2<f64>,
        tetra: Vec<usize>,
    ) -> Vec<usize> {
        if Self::signed_volume(vertices, tetra[0], tetra[1], tetra[2], tetra[3]) <= 0.0 {
            println!("new tetra: {:?}", tetra);
            return vec![tetra[0], tetra[3], tetra[2], tetra[1]]
        } else {
            return tetra
        }

    }
    fn make_cube_mesh() -> (Array2<f64>, Array2<usize>) {
        let verts = array![[0., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.],
                           [1., 0., 0.],
                           [1., 1., 0.],
                           [1., 0., 1.],
                           [1., 1., 1.]];

        let tetras = array![[0,1,2,3],
                            [1,4,6,5]]; 
        (verts, tetras)
    }

    fn make_beam_mesh(
        width : f64,
        height : f64,
        length : f64,
        shape : (usize, usize, usize)
    ) -> (Array2<f64>, Array2<usize>) {
        if width < 0.0 || height < 0.0 || length < 0.0 { panic!("Could not create mesh! Width/height cannot be negative") }

        let (nx, ny, nz) = shape;

        let dx = width  / nx as f64;
        let dy = height / ny as f64;
        let dz = length / nz as f64;
        
        let x0 = -width  / 2.0;
        let y0 = -height / 2.0;
        let z0 = -length / 2.0;
        
        let nverts = (nx + 1) * (ny + 1) * (nz + 1);
        let mut verts = Array2::<f64>::zeros((nverts, 3));
        
        // build vertices
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let idx = i + j*(nx+1) + k*(nx+1)*(ny+1);
        
                    verts[[idx, 0]] = x0 + i as f64 * dx;
                    verts[[idx, 1]] = y0 + j as f64 * dy;
                    verts[[idx, 2]] = z0 + k as f64 * dz;
                }
            }
        }

        // index helper
        let idx = |i: usize, j: usize, k: usize| -> usize {
            i + j*(nx+1) + k*(nx+1)*(ny+1)
        };

        // build tetraheda
        let ncells = nx * ny * nz;
        let ntets = 6 * ncells;
        
        let mut tets = Array2::<usize>::zeros((ntets, 4));
        
        let mut e = 0;
        
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
        
                    let v000 = idx(i,   j,   k);
                    let v100 = idx(i+1, j,   k);
                    let v010 = idx(i,   j+1, k);
                    let v110 = idx(i+1, j+1, k);
        
                    let v001 = idx(i,   j,   k+1);
                    let v101 = idx(i+1, j,   k+1);
                    let v011 = idx(i,   j+1, k+1);
                    let v111 = idx(i+1, j+1, k+1);
        
                    // 6 tetrahedra
                    tets.slice_mut(s![e, ..]).assign(&array![v000, v100, v010, v001]); e += 1;
                    tets.slice_mut(s![e, ..]).assign(&array![v100, v110, v010, v111]); e += 1;
                    tets.slice_mut(s![e, ..]).assign(&array![v100, v010, v001, v111]); e += 1;
                    tets.slice_mut(s![e, ..]).assign(&array![v010, v001, v011, v111]); e += 1;
                    tets.slice_mut(s![e, ..]).assign(&array![v100, v001, v101, v111]); e += 1;
                    tets.slice_mut(s![e, ..]).assign(&array![v001, v011, v101, v111]); e += 1;
                }
            }
        } 
        
        // return tuple of vertices and tetraheda
        (verts, tets)
    }
    
    fn compute_tetrahedon_volumes(vertices: &Array2<f64>, tetraheda: &Array2<usize>) -> Array1<f64> {
        tetraheda
            .outer_iter()
            .map(|tetra| {
                Self::signed_volume(vertices, tetra[0], tetra[1], tetra[2], tetra[3])
            })
            .collect()
    }
    fn compute_vertex_tetrahedon_adjacency(vertices: &Array2<f64>, tetraheda: &Array2<usize>) -> Vec<Vec<usize>> {
        let vertex_count: usize = vertices.shape()[0];
        let mut vertex_neighbor_tetras: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];

        for (tetra_idx, tetra) in tetraheda.outer_iter().enumerate() {
            for &v in tetra.iter() { 
                vertex_neighbor_tetras[v].push(tetra_idx);
            }
        }
        vertex_neighbor_tetras
    }
    
}
