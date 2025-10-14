mod mesh;
mod cv;
mod sim;  

use bevy::{ 
    prelude::*,
    window::WindowResolution,
};
use bevy::camera::Camera2d;
use bevy::mesh::{Mesh, PrimitiveTopology, Indices, VertexAttributeValues};
use bevy::asset::RenderAssetUsages;
use bevy::sprite_render::{Wireframe2dPlugin, Wireframe2dConfig};
use sim::cauchy_fvm::CauchyFVM;

fn main() {
    let tmesh = mesh::TriangleMesh::new_beam(6.0, 2.0, (12, 4));
    App::new()
        .add_plugins((DefaultPlugins
                .set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(640,480).with_scale_factor_override(1.0),
                    ..default() 
                }),
                ..default()
            }),
            Wireframe2dPlugin::default()))
        .insert_resource(TriangleMeshResource(tmesh))
        .insert_resource(Time::<Fixed>::from_hz(1200.0))
        .add_systems(Startup, beam)
        .add_systems(Startup, create_simulator)
        .add_systems(FixedUpdate, update_simulator)
        .add_systems(Update, set_new_vertices_with_simulator)
        .add_systems(Update, toggle_wireframe)
        .run(); 
}

fn beam(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    beam_mesh: Res<TriangleMeshResource>,
) {
    // create Bevy Mesh
    let mut bevy_beam = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(), 
    );

    // add vertex positions from TriangleMesh struct
    let mut v_pos: Vec<[f32; 3]> = Vec::new();
    for vertex in beam_mesh.0.vertices.rows() {
        let x = vertex[[0]] as f32;
        let y = vertex[[1]] as f32;
        let z = 0.0;
        v_pos.push([x,y,z]);
    }
    bevy_beam.insert_attribute(Mesh::ATTRIBUTE_POSITION, v_pos.clone());
    
    // triangle list
    let mut indices: Vec<u32> = Vec::new();
    for triangle in beam_mesh.0.triangles.rows() {
        let i = triangle[[0]] as u32;
        let j = triangle[[1]] as u32;
        let k = triangle[[2]] as u32;
        indices.extend_from_slice(&[i,j,k]);
    }
    bevy_beam.insert_indices(Indices::U32(indices));
    
    let beam_ptr = meshes.add(bevy_beam);
    
    // set beam mesh color
    let color = Color::srgb(1., 0., 0.); 

    commands.spawn((
            Mesh2d(beam_ptr),
            MeshMaterial2d(materials.add(color)),
            ));

    commands.spawn((
        Camera2d,
        Projection::from(OrthographicProjection {
            scale: 0.02,
            ..OrthographicProjection::default_2d()
            },
        )
    ));
}

#[derive(Resource)]
struct TriangleMeshResource(mesh::TriangleMesh);

#[derive(Component)]
pub struct MeshSimulator {
    // wraps the CauchyFVM simulator
    sim: CauchyFVM,
}

fn create_simulator(mut commands: Commands, tmesh: Res<TriangleMeshResource>) {
    let simulator = MeshSimulator { 
        sim: CauchyFVM::new(&tmesh.0, "rubber", 1e-3) 
    };
    commands.spawn(simulator);
}

fn update_simulator(
    mut query: Query<&mut MeshSimulator>) {
        for mut simulator in &mut query {
            simulator.sim.update();
        }
}

fn set_new_vertices_with_simulator(query: Query<&MeshSimulator>, shape: Single<&Mesh2d>, mut meshes: ResMut<Assets<Mesh>>) {
    let Some(mesh) = meshes.get_mut(*shape) else { return; };
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        let simulator = query.single();
        let sim_vertices = &simulator.unwrap().sim.sim_mesh.vertices;

        for (idx, position) in positions.iter_mut().enumerate() {
            let v = sim_vertices.row(idx);
            position[0] = v[0] as f32;
            position[1] = v[1] as f32;
        }
    }
}

fn toggle_wireframe(
    mut wireframe_config: ResMut<Wireframe2dConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        wireframe_config.global = !wireframe_config.global;
    }
}
