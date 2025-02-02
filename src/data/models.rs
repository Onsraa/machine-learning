use crate::algorithms::linear_regression::prepare_data;
use crate::params::*;
use bevy::color::palettes::css as color;
use bevy::prelude::{Transform, *};
use nalgebra::*;
use super::data_model::*;

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
pub struct DatasetConverter {
    pub inputs: DMatrix<f64>,
    pub outputs: DVector<f64>,
    pub is_classification: bool,
}

impl DatasetConverter {
    // Convertit les données nalgebra en points Bevy
    pub fn to_bevy_points(&self) -> Vec<Point> {
        let mut points = Vec::new();

        // Pour chaque point dans les données
        for i in 0..self.inputs.nrows() {
            let x = self.inputs[(i, 0)] as f32;
            let y = if self.inputs.ncols() > 1 {
                self.inputs[(i, 1)] as f32
            } else {
                self.outputs[i] as f32
            };
            let z = if self.inputs.ncols() > 2 {
                self.inputs[(i, 2)] as f32
            } else {
                0.0
            };

            // Déterminer la couleur
            let color = if self.is_classification {
                match self.outputs[i] as i32 {
                    0 => Color::from(color::BLUE),
                    1 => Color::from(color::RED),
                    2 => Color::from(color::GREEN),
                    _ => Color::from(color::BLACK),
                }
            } else {
                Color::from(color::BLACK)
            };

            points.push(Point(x, y, z, color));
        }

        points
    }

    // Convertit les points Bevy en données nalgebra
    pub fn from_bevy_points(points: &[Point], is_classification: bool) -> Self {
        let n_points = points.len();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for Point(x, y, z, color) in points {
            if *z == 0.0 {
                // Cas 2D
                inputs.push(vec![*x as f64]);
                outputs.push(*y as f64);
            } else {
                // Cas 3D
                inputs.push(vec![*x as f64, *y as f64]);
                outputs.push(*z as f64);
            }

            if is_classification {
                // Convertir les couleurs en classes
                outputs.push(match color {
                    c if *c == Color::from(color::BLUE) => 0.0,
                    c if *c == Color::from(color::RED) => 1.0,
                    c if *c == Color::from(color::GREEN) => 2.0,
                    _ => 0.0,
                });
            }
        }

        let n_features = inputs[0].len();
        let x_matrix = DMatrix::from_fn(n_points, n_features, |i, j| inputs[i][j]);
        let y_vector = DVector::from_vec(outputs);

        DatasetConverter {
            inputs: x_matrix,
            outputs: y_vector,
            is_classification,
        }
    }
}

// Fonction de mise à jour adaptée
pub fn update_points(
    mut points: ResMut<Points>,
    mut dataset: ResMut<DatasetConverter>,
    data_model: Res<DataModel>,
) {
    let generated_dataset = generate_dataset(&data_model);
    let (inputs_matrix, outputs_vector) =
        prepare_data(generated_dataset.inputs, generated_dataset.outputs);

    *dataset = DatasetConverter {
        inputs: inputs_matrix,
        outputs: outputs_vector,
        is_classification: matches!(
            *data_model,
            DataModel::LinearSimple
                | DataModel::LinearMultiple
                | DataModel::XOR
                | DataModel::Cross
                | DataModel::MultiLinear3Classes
                | DataModel::MultiCross
        ),
    };

    points.data = dataset.to_bevy_points();
}

#[derive(Resource, Default)]
pub struct MaxCoordinates {
    pub x: f32,
    pub y: f32,
    pub z: f32,
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

pub fn update_max_coordinates(mut points: ResMut<Points>, mut max_coords: ResMut<MaxCoordinates>) {
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
    let scale_factor = AXIS_LENGTH / 2.0 * 0.9;

    for Point(x, y, z, color) in points.data.iter() {
        let mesh = meshes.add(Sphere::new(0.03).mesh().uv(32, 18));
        let material = materials.add(StandardMaterial {
            base_color: *color,
            ..Default::default()
        });

        let (pos_x, pos_y, pos_z) = if data_model.is_normalized() {
            (*x * scale_factor, *y * scale_factor, *z * scale_factor)
        } else {
            let normalized_x = x / max_coords.x;
            let normalized_y = y / max_coords.y;
            let normalized_z = if max_coords.z != 0.0 {
                z / max_coords.z
            } else {
                0.0
            };

            (
                normalized_x * scale_factor,
                normalized_y * scale_factor,
                normalized_z * scale_factor,
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
