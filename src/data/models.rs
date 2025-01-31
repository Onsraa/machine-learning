use crate::systems::graph::AXIS_LENGTH;
use bevy::color::palettes::css as color;
use bevy::prelude::*;
use rand::Rng;

pub struct Point(pub f32, pub f32, pub f32, pub Color); // x, y, z, color


#[derive(Component)]
pub struct DataPoint;

#[derive(Resource, Default)]
pub struct Points {
    pub data: Vec<Point>,
}

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelState {
    Ready,
    #[default]
    Updating,
}

#[derive(Resource, Default)]
pub enum DataModel {
    #[default]
    LinearSimple,
    LinearMultiple,
    XOR,
    Cross,
    MultiLinear3Classes,
    MultiCross,
    LinearSimple2d,
    LinearSimple3d,
    LinearTricky3d,
    NonLinearSimple2d,
    NonLinearSimple3d,
}

impl DataModel {
    pub fn is_normalized(&self) -> bool {
        matches!(
            self,
            DataModel::XOR
                | DataModel::Cross
                | DataModel::MultiLinear3Classes
                | DataModel::MultiCross
        )
    }
}

#[derive(Resource, Default)]
pub struct MaxCoordinates {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub fn update_points(mut points: ResMut<Points>, data_model: Res<DataModel>) {
    points.data = match *data_model {
        DataModel::LinearSimple => create_linear_simple_model(),
        DataModel::LinearMultiple => create_linear_multiple_model(),
        DataModel::XOR => create_xor_model(),
        DataModel::Cross => create_cross_model(),
        DataModel::MultiLinear3Classes => create_multi_linear_3_classes_model(),
        DataModel::MultiCross => create_multi_cross_model(),
        DataModel::LinearSimple2d => create_linear_simple_2d_model(),
        DataModel::LinearSimple3d => create_linear_simple_3d_model(),
        DataModel::LinearTricky3d => create_linear_tricky_3d_model(),
        DataModel::NonLinearSimple2d => create_non_linear_simple_2d_model(),
        DataModel::NonLinearSimple3d => create_non_linear_simple_3d_model(),
    };
}

pub fn find_max_coordinates(points: &Vec<Point>) -> (f32, f32, f32) {
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut max_z = f32::MIN;

    for Point(x, y, z, _) in points.iter() {
        max_x = max_x.max(x.abs());
        max_y = max_y.max(y.abs());
        max_z = max_z.max(z.abs());
    }

    (max_x, max_y, max_z)
}

pub fn update_max_coordinates(
    mut points: ResMut<Points>,
    mut max_coords: ResMut<MaxCoordinates>,
) {
    let (max_x, max_y, max_z) = find_max_coordinates(&points.data);
    max_coords.x = max_x;
    max_coords.y = max_y;
    max_coords.z = max_z;
}

pub fn draw_points(
    mut commands: Commands,
    mut points: ResMut<Points>,
    mut max_coords: ResMut<MaxCoordinates>,
    data_model: Res<DataModel>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut data_points: Query<Entity, With<DataPoint>>,
    mut next_state: ResMut<NextState<ModelState>>,
) {
    for point in data_points.iter() {
        commands.entity(point).despawn();
    }
    let scale_factor = AXIS_LENGTH / 2.0 * 0.8;

    for Point(x, y, z, color) in points.data.iter() {
        let mesh = meshes.add(Sphere::new(0.05).mesh().uv(32, 18));
        let material = materials.add(StandardMaterial {
            base_color: *color,
            ..Default::default()
        });

        let (pos_x, pos_y, pos_z) = if data_model.is_normalized() {
            (*x * scale_factor, *y * scale_factor, *z * scale_factor)
        } else {
            let normalized_x = x / max_coords.x;
            let normalized_y = y / max_coords.y;
            let normalized_z = if max_coords.z != 0.0 { z / max_coords.z } else { 0.0 };

            (
                normalized_x * scale_factor,
                normalized_y * scale_factor,
                normalized_z * scale_factor
            )
        };

        commands.spawn((
            DataPoint,
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(pos_x, pos_y, pos_z),
        ));
    }
    next_state.set(ModelState::Ready)
}

fn create_linear_simple_model() -> Vec<Point> {
    vec![
        Point(1.0, 1.0, 0.0, Color::from(color::BLUE)),
        Point(2.0, 3.0, 0.0, Color::from(color::RED)),
        Point(3.0, 3.0, 0.0, Color::from(color::RED)),
    ]
}

fn create_linear_multiple_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(100);

    for _ in 0..50 {
        let x = rng.gen::<f32>() * 0.9 + 1.0;
        let y = rng.gen::<f32>() * 0.9 + 1.0;
        pts.push(Point(x, y, 0.0, Color::from(color::BLUE)));
    }

    for _ in 0..50 {
        let x = rng.gen::<f32>() * 0.9 + 2.0;
        let y = rng.gen::<f32>() * 0.9 + 2.0;
        pts.push(Point(x, y, 0.0, Color::from(color::RED)));
    }

    pts
}

fn create_xor_model() -> Vec<Point> {
    vec![
        Point(1.0, 0.0, 0.0, Color::from(color::BLUE)),
        Point(0.0, 1.0, 0.0, Color::from(color::BLUE)),
        Point(0.0, 0.0, 0.0, Color::from(color::RED)),
        Point(1.0, 1.0, 0.0, Color::from(color::RED)),
    ]
}

fn create_cross_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(500);

    for _ in 0..500 {
        let x: f32 = rng.gen_range(-1.0..1.0);
        let y: f32 = rng.gen_range(-1.0..1.0);
        let label = if x.abs() <= 0.3 || y.abs() <= 0.3 {
            1
        } else {
            -1
        };
        let color = if label == 1 {
            Color::from(color::BLUE)
        } else {
            Color::from(color::RED)
        };
        pts.push(Point(x, y, 0.0, color));
    }
    pts
}

fn create_multi_linear_3_classes_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::new();

    for _ in 0..500 {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);

        let c1 = -x - y - 0.5 > 0.0 && y < 0.0 && (x - y - 0.5) < 0.0;
        let c2 = -x - y - 0.5 < 0.0 && y > 0.0 && (x - y - 0.5) < 0.0;
        let c3 = -x - y - 0.5 < 0.0 && y < 0.0 && (x - y - 0.5) > 0.0;

        let color = if c1 {
            Some(Color::from(color::BLUE))
        } else if c2 {
            Some(Color::from(color::RED))
        } else if c3 {
            Some(Color::from(color::GREEN))
        } else {
            None
        };

        if let Some(col) = color {
            pts.push(Point(x, y, 0.0, col));
        }
    }

    pts
}

fn create_multi_cross_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(1000);

    for _ in 0..1000 {
        let x: f32 = rng.gen_range(-1.0..1.0);
        let y: f32 = rng.gen_range(-1.0..1.0);

        let cond_blue: f32 = x % 0.5;
        let cond_red: f32 = y % 0.5;

        let cond_blue_ok = cond_blue.abs() <= 0.25 && cond_red.abs() > 0.25;
        let cond_red_ok = cond_blue.abs() > 0.25 && cond_red.abs() <= 0.25;

        let color = if cond_blue_ok {
            Color::from(color::BLUE)
        } else if cond_red_ok {
            Color::from(color::RED)
        } else {
            Color::from(color::GREEN)
        };

        pts.push(Point(x, y, 0.0, color));
    }

    pts
}

fn create_linear_simple_2d_model() -> Vec<Point> {
    vec![
        Point(1.0, 2.0, 0.0, Color::BLACK),
        Point(2.0, 3.0, 0.0, Color::BLACK),
    ]
}

fn create_linear_simple_3d_model() -> Vec<Point> {
    vec![
        Point(1.0, 1.0, 2.0, Color::from(color::BLACK)),
        Point(2.0, 2.0, 3.0, Color::from(color::BLACK)),
        Point(3.0, 1.0, 2.5, Color::from(color::BLACK)),
    ]
}

fn create_linear_tricky_3d_model() -> Vec<Point> {
    vec![
        Point(1.0, 1.0, 1.0, Color::from(color::BLACK)),
        Point(2.0, 2.0, 2.0, Color::from(color::BLACK)),
        Point(3.0, 3.0, 3.0, Color::from(color::BLACK)),
    ]
}

fn create_non_linear_simple_2d_model() -> Vec<Point> {
    vec![
        Point(1.0, 2.0, 0.0, Color::BLACK),
        Point(2.0, 3.0, 0.0, Color::BLACK),
        Point(3.0, 2.5, 0.0, Color::BLACK),
    ]
}

fn create_non_linear_simple_3d_model() -> Vec<Point> {
    vec![
        Point(1.0, 0.0, 2.0, Color::from(color::BLACK)),
        Point(0.0, 1.0, 1.0, Color::from(color::BLACK)),
        Point(1.0, 1.0, -2.0, Color::from(color::BLACK)),
        Point(0.0, 0.0, -1.0, Color::from(color::BLACK)),
    ]
}
