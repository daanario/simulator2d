mod sim_core;
mod render3d;

use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use bevy::sprite_render::{Wireframe2dPlugin};
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use sim_core::mesh;
use render3d::*;

fn main() {
    /*
    let tmesh = sim_core::mesh3d::TetrahedalMesh::new_beam(
        6.0, 2.0, 2.0, (12, 4, 4)
    );
    */
    let tmesh = sim_core::mesh3d::TetrahedalMesh::new_cube();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::new(1280, 720)
                    .with_scale_factor_override(1.0),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(PanOrbitCameraPlugin)
        .insert_resource(TetraMeshResource(tmesh))
        .add_systems(Startup, beam_3d)
        .add_plugins(WireframePlugin::default())
        .run();
}
