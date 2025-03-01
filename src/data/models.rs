use crate::data::*;
use crate::params::*;
use bevy::color::palettes::css as color;
use bevy::prelude::{Transform, *};
use nalgebra::*;

pub fn prepare_data(inputs: Vec<Vec<f64>>, outputs: Vec<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let n_samples = inputs.len();
    let n_features = inputs[0].len();
    let x_matrix = DMatrix::from_fn(n_samples, n_features, |i, j| inputs[i][j]);
    let y_vector = DVector::from_vec(outputs);
    (x_matrix, y_vector)
}

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

#[derive(Resource)]
pub struct DatasetConverter {
    pub inputs: DMatrix<f64>,
    pub outputs: DVector<f64>,
    pub is_classification: bool,
}

impl Default for DatasetConverter {
    fn default() -> Self {
        Self {
            inputs: DMatrix::zeros(0, 0),
            outputs: DVector::zeros(0),
            is_classification: false,
        }
    }
}

impl DatasetConverter {
    pub fn to_bevy_points(&self) -> Vec<Point> {
        let mut points = Vec::new();

        for i in 0..self.inputs.nrows() {
            let (x, y, z) = match self.inputs.ncols() {
                // Cas 2D (régression)
                1 => {
                    let x = self.inputs[(i, 0)] as f32;
                    let y = self.outputs[i] as f32;
                    (x, y, 0.0)
                }
                // Cas 2D (classification) ou 3D (régression)
                2 => {
                    if self.is_classification {
                        // Classification 2D
                        let x = self.inputs[(i, 0)] as f32;
                        let y = self.inputs[(i, 1)] as f32;
                        (x, y, 0.0)
                    } else {
                        // Régression 3D
                        let x = self.inputs[(i, 0)] as f32;
                        let y = self.inputs[(i, 1)] as f32;
                        let z = self.outputs[i] as f32;
                        (x, y, z)
                    }
                }
                _ => {
                    let x = self.inputs[(i, 0)] as f32;
                    let y = self.inputs[(i, 1)] as f32;
                    let z = self.outputs[i] as f32;
                    (x, y, z)
                }
            };

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

    pub fn from_bevy_points(points: &[Point], is_classification: bool) -> Self {
        let n_points = points.len();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for Point(x, y, z, _) in points {
            if is_classification {
                inputs.push(vec![*x as f64, *y as f64]);
            } else if *z == 0.0 {
                inputs.push(vec![*x as f64]);
                outputs.push(*y as f64);
            } else {
                inputs.push(vec![*x as f64, *y as f64]);
                outputs.push(*z as f64);
            }
        }

        let n_features = inputs[0].len();
        let x_matrix = DMatrix::from_fn(n_points, n_features, |i, j| inputs[i][j]);
        let y_vector = if is_classification {
            DVector::from_vec(vec![0.0; n_points])
        } else {
            DVector::from_vec(outputs)
        };

        DatasetConverter {
            inputs: x_matrix,
            outputs: y_vector,
            is_classification,
        }
    }
}

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

pub fn update_max_coordinates(points: Res<Points>, mut max_coords: ResMut<MaxCoordinates>) {
    let (max_x, max_y, max_z) = find_max_coordinates(&points.data);
    max_coords.x = max_x;
    max_coords.y = max_y;
    max_coords.z = max_z;
}

pub fn draw_points(
    mut commands: Commands,
    points: Res<Points>,
    max_coords: Res<MaxCoordinates>,
    data_model: Res<DataModel>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    data_points: Query<Entity, With<DataPoint>>,
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
