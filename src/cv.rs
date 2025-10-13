use ndarray::prelude::*;
use ndarray::{Array1, Array2};
//use ndarray_linalg;
use crate::mesh::TriangleMesh;
use ndarray_linalg::Norm;




pub struct MedianCentroidControlVolume {
    pub vertex: usize,
    pub neighbor_tri_ids: Vec<usize>,
    pub neighbor_triangles: Array2<usize>,
    
    pub lij: Array1<f64>, // lengths of edge (i,j) for each element
    pub lik: Array1<f64>, // lengths of edge (i,k) for each element
    pub nij: Array2<f64>, // unit normals for edge (i,j) of each element
    pub nik: Array2<f64>, // unit normals for edge (i,j) of each element

    pub area: f64, // area of the entire control volume
}

impl MedianCentroidControlVolume {
    pub fn new(vertex: usize, mesh: &TriangleMesh) -> MedianCentroidControlVolume {
        let neighbor_tri_ids = &mesh.vertex_neighbor_tris[vertex];
        
        let mut neighbor_triangles: Array2<usize> = Array2::zeros((neighbor_tri_ids.len(), 3));
        
        // ensure that vertex is always the first index i for every triangle
        for (idx, tri_idx) in neighbor_tri_ids.iter().enumerate() {
            let triangle = &mesh.triangles.slice(s![*tri_idx as usize, ..]);
            neighbor_triangles.slice_mut(s![idx, ..]).assign(&Self::get_cyclic_permutation(vertex, triangle));
        }
        
        // compute lengths and normals for each triangle element
        let mut lij: Array1<f64> = Array1::zeros(neighbor_tri_ids.len());
        let mut lik: Array1<f64> = Array1::zeros(neighbor_tri_ids.len());
        let mut nij: Array2<f64> = Array2::zeros((neighbor_tri_ids.len(), 2));
        let mut nik: Array2<f64> = Array2::zeros((neighbor_tri_ids.len(), 2));

        for (idx, triangle) in neighbor_triangles.outer_iter().enumerate() {
            
            let i = triangle[[0]];
            let j = triangle[[1]];
            let k = triangle[[2]];
            
            let pi = &mesh.vertices.slice(s![i, ..]); // vertex i 
            let pj = &mesh.vertices.slice(s![j, ..]); // vertex j 
            let pk = &mesh.vertices.slice(s![k, ..]); // vertex k
     
            let eij = pj - pi; // edge between vertices i and j
            let eik = pk - pi; // edge between vertices k and i
            
            lij.slice_mut(s![idx]).fill(eij.norm());
            lik.slice_mut(s![idx]).fill(eik.norm());
            
            // hat = (-y, x)
            // Nij = -hat(Eij)/Lij
            // Nik = hat(Eik)/Lik

            nij.slice_mut(s![idx, ..]).assign(&(-array![-eij[1], eij[0]] / lij[idx]));
            nik.slice_mut(s![idx, ..]).assign(&(array![-eik[1], eik[0]] / lik[idx])); 
            
        }
        
        let element_areas = &mesh.areas.select(Axis(0), neighbor_tri_ids);
        let area: f64 = element_areas.sum();

        MedianCentroidControlVolume {
            vertex,
            neighbor_tri_ids: neighbor_tri_ids.to_vec(),
            neighbor_triangles,
            lij,
            lik,
            nij,
            nik,
            area,
        }
    } 

    fn get_cyclic_permutation(vertex: usize, triangle: &ArrayView1<usize>) -> Array1<usize> {
        let i = triangle[[0]];
        let j = triangle[[1]];
        let k = triangle[[2]];
        if vertex == i { return array![i, j, k]; }
        else if vertex == j { return array![j, k, i]; }
        else if vertex == k { return array![k, i, j]; }

        panic!("Control Volume error! Could not get cyclic permutation of triangle")
    }
}
