use crate::sim_core;

use bevy::{ 
    prelude::*,
};
use bevy::camera::Camera2d;
use bevy::mesh::{Mesh, PrimitiveTopology, Indices, VertexAttributeValues};
use bevy::asset::RenderAssetUsages;
use bevy_panorbit_camera::PanOrbitCamera;
use bevy::sprite_render::{Wireframe2d, Wireframe2dColor};
use bevy::pbr::wireframe::Wireframe;
use sim_core::cauchy_fvm::CauchyFVM;
use sim_core::mesh;
use ndarray::Array2;

use std::collections::HashMap;

pub fn beam_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    beam_mesh: Res<TetraMeshResource>,
) {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(),
    );

    // vertices
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for v in beam_mesh.0.vertices.rows() {
        positions.push([v[0] as f32, v[1] as f32, v[2] as f32]);
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    let mut indices: Vec<u32> = Vec::new();

    for tet in beam_mesh.0.tetraheda.rows() {
        let i = tet[0] as u32;
        let j = tet[1] as u32;
        let k = tet[2] as u32;
        let l = tet[3] as u32;
    
        // 4 faces per tetrahedron
        indices.extend_from_slice(&[i, j, k]);
        indices.extend_from_slice(&[i, j, l]);
        indices.extend_from_slice(&[i, k, l]);
        indices.extend_from_slice(&[j, k, l]);
    }

    mesh.insert_indices(Indices::U32(indices));

    let mesh_handle = meshes.add(mesh);

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::BLACK,
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Wireframe,
    ));

    // light
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 5.0, 5.0),
    ));

    // camera
    /*
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 6.0, 8.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
    ));
    */
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera::default(),
    ));
}

#[derive(Resource)]
pub struct TetraMeshResource(pub sim_core::mesh3d::TetrahedalMesh);

