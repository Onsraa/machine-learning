use bevy::color::palettes::css::*;
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use std::f32::consts::PI;
use bevy::render::primitives::Aabb;

const AXIS_LENGTH: f32 = 4.0;
const AXIS_THICKNESS: f32 = 0.03;
pub struct Graph3DPlugin;

impl Plugin for Graph3DPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_graph);
        //app.add_systems(Update, draw_axes);
    }
}

#[derive(Component)]
struct ShowAxes;

fn draw_axes(mut gizmos: Gizmos, query: Query<(&Transform, &Aabb), With<ShowAxes>>) {
    for (&transform, &aabb) in &query {
        let length = aabb.half_extents.length();
        gizmos.axes(transform, length);
    }
}

#[derive(Component)]
struct Axis;

#[derive(Component)]
enum AxisType {
    X,
    Y,
    Z,
}

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
            AxisType::X,
            ShowAxes
        ))
        .id();

    let x_cone = commands
        .spawn((
            Mesh3d(meshes.add(Cone::new(AXIS_THICKNESS * 2., AXIS_THICKNESS * 5.))),
            MeshMaterial3d(red_material.clone()),
            Transform::from_xyz(0.0, AXIS_LENGTH / 2.0, 0.0),
            ShowAxes
        ))
        .id();

    commands.entity(x_axis).add_child(x_cone);

    let y_axis = commands
        .spawn((
            Mesh3d(meshes.add(Cylinder::new(AXIS_THICKNESS / 2., AXIS_LENGTH))),
            MeshMaterial3d(green_material.clone()),
            Transform::default(),
            Axis,
            AxisType::Y,
            ShowAxes
        ))
        .id();

    let y_cone = commands
        .spawn((
            Mesh3d(meshes.add(Cone::new(AXIS_THICKNESS * 2., AXIS_THICKNESS * 5.))),
            MeshMaterial3d(green_material.clone()),
            Transform::from_xyz(0.0, AXIS_LENGTH / 2.0, 0.0),
            ShowAxes
        ))
        .id();

    commands.entity(y_axis).add_child(y_cone);

    let z_axis = commands
        .spawn((
            Mesh3d(meshes.add(Cylinder::new(AXIS_THICKNESS / 2., AXIS_LENGTH))),
            MeshMaterial3d(blue_material.clone()),
            Transform::default().with_rotation(Quat::from_rotation_x(PI / 2.)),
            Axis,
            AxisType::Z,
            ShowAxes
        ))
        .id();

    let z_cone = commands
        .spawn((
            Mesh3d(meshes.add(Cone::new(AXIS_THICKNESS * 2., AXIS_THICKNESS * 5.))),
            MeshMaterial3d(blue_material.clone()),
            Transform::from_xyz(0.0, AXIS_LENGTH / 2.0, 0.0),
            ShowAxes
        ))
        .id();

    commands.entity(z_axis).add_child(z_cone);

    // Lumière
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

fn update(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Axis>>,
) {
    let delta = time.delta_secs();
    for mut axis in query.iter_mut() {
        axis.rotation.z += 0.1 * delta;
    }
}
