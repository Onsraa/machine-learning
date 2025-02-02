use bevy::asset::Assets;
use bevy::color::palettes::basic::{BLUE, GREEN, RED};
use bevy::math::Quat;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
use std::f32::consts::PI;
use crate::params::*;

#[derive(Component)]
struct Axis;

pub fn setup_graph(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let red_material = materials.add(StandardMaterial {
        base_color: RED.into(),
        ..Default::default()
    });

    let blue_material = materials.add(StandardMaterial {
        base_color: BLUE.into(),
        ..Default::default()
    });

    let green_material = materials.add(StandardMaterial {
        base_color: GREEN.into(),
        ..Default::default()
    });

    let x_axis = commands
        .spawn((
            Mesh3d(meshes.add(Cylinder::new(AXIS_THICKNESS / 2., AXIS_LENGTH))),
            MeshMaterial3d(red_material.clone()),
            Transform::default().with_rotation(Quat::from_rotation_z(-PI / 2.)),
            Axis,
        ))
        .id();

    let x_cone = commands
        .spawn((
            Mesh3d(meshes.add(Cone::new(AXIS_THICKNESS * 2., AXIS_THICKNESS * 5.))),
            MeshMaterial3d(red_material.clone()),
            Transform::from_xyz(0.0, AXIS_LENGTH / 2.0, 0.0),
        ))
        .id();

    commands.entity(x_axis).add_child(x_cone);

    let y_axis = commands
        .spawn((
            Mesh3d(meshes.add(Cylinder::new(AXIS_THICKNESS / 2., AXIS_LENGTH))),
            MeshMaterial3d(green_material.clone()),
            Transform::default(),
            Axis,
        ))
        .id();

    let y_cone = commands
        .spawn((
            Mesh3d(meshes.add(Cone::new(AXIS_THICKNESS * 2., AXIS_THICKNESS * 5.))),
            MeshMaterial3d(green_material.clone()),
            Transform::from_xyz(0.0, AXIS_LENGTH / 2.0, 0.0),
        ))
        .id();

    commands.entity(y_axis).add_child(y_cone);

    let z_axis = commands
        .spawn((
            Mesh3d(meshes.add(Cylinder::new(AXIS_THICKNESS / 2., AXIS_LENGTH))),
            MeshMaterial3d(blue_material.clone()),
            Transform::default().with_rotation(Quat::from_rotation_x(PI / 2.)),
            Axis,
        ))
        .id();

    let z_cone = commands
        .spawn((
            Mesh3d(meshes.add(Cone::new(AXIS_THICKNESS * 2., AXIS_THICKNESS * 5.))),
            MeshMaterial3d(blue_material.clone()),
            Transform::from_xyz(0.0, AXIS_LENGTH / 2.0, 0.0),
        ))
        .id();

    commands.entity(z_axis).add_child(z_cone);
}
