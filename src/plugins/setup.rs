use crate::components::camera::*;
use crate::systems::camera::*;
use crate::systems::graph::setup_graph;
use bevy::prelude::*;

pub struct SetupPlugin;

impl Plugin for SetupPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraSettings>();
        app.add_systems(Startup, setup);
        app.add_systems(Startup, setup_graph);
        app.add_systems(Update, orbit);
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            intensity: 10_000_000.,
            range: 100.0,
            shadow_depth_bias: 0.2,
            ..default()
        },
        Transform::from_xyz(8.0, 8.0, 8.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.0, 4.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
