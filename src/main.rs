mod sim_core;
mod render;

use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy::sprite_render::{Wireframe2dPlugin};
use sim_core::mesh;
use render::*;

fn main() {
    let tmesh = mesh::TriangleMesh::new_beam(6.0, 2.0, (12, 4));
    App::new()
        .add_plugins((DefaultPlugins
                .set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(1280,720).with_scale_factor_override(1.0),
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
        .run(); 
}
