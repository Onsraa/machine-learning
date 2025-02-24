use crate::data::universal_dataset::TaskType;
use crate::data::Point;
use nalgebra::{DMatrix, DVector};

pub struct DatasetConverter {
    pub inputs: DMatrix<f64>,
    pub targets: DVector<f64>,
    pub task: TaskType,
}

impl DatasetConverter {
    pub fn to_bevy_points(&self) -> Vec<Point> {
        let mut points = Vec::new();
        for i in 0..self.inputs.nrows() {
            let (x, y, z) = match self.inputs.ncols() {
                1 => {
                    let x = self.inputs[(i, 0)] as f32;
                    let y = self.targets[i] as f32;
                    (x, y, 0.0)
                }
                2 => {
                    if self.task == TaskType::Classification {
                        let x = self.inputs[(i, 0)] as f32;
                        let y = self.inputs[(i, 1)] as f32;
                        (x, y, 0.0)
                    } else {
                        let x = self.inputs[(i, 0)] as f32;
                        let y = self.inputs[(i, 1)] as f32;
                        let z = self.targets[i] as f32;
                        (x, y, z)
                    }
                }
                _ => {
                    let x = self.inputs[(i, 0)] as f32;
                    let y = self.inputs[(i, 1)] as f32;
                    let z = self.targets[i] as f32;
                    (x, y, z)
                }
            };
            let color = if self.task == TaskType::Classification {
                match self.targets[i] as i32 {
                    0 => bevy::prelude::Color::from(bevy::color::palettes::css::BLUE),
                    1 => bevy::prelude::Color::from(bevy::color::palettes::css::RED),
                    2 => bevy::prelude::Color::from(bevy::color::palettes::css::GREEN),
                    _ => bevy::prelude::Color::from(bevy::color::palettes::css::BLACK),
                }
            } else {
                bevy::prelude::Color::from(bevy::color::palettes::css::BLACK)
            };
            points.push(Point(x, y, z, color));
        }
        points
    }

    pub fn from_bevy_points(points: &[Point], task: TaskType) -> Self {
        let n_points = points.len();
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for Point(x, y, z, _) in points {
            match task {
                TaskType::Classification => {
                    inputs.push(vec![*x as f64, *y as f64]);
                }
                TaskType::Regression => {
                    if *z == 0.0 {
                        inputs.push(vec![*x as f64]);
                        targets.push(*y as f64);
                    } else {
                        inputs.push(vec![*x as f64, *y as f64]);
                        targets.push(*z as f64);
                    }
                }
            }
        }
        let n_features = inputs[0].len();
        let x_matrix = DMatrix::from_fn(n_points, n_features, |i, j| inputs[i][j]);
        let y_vector = match task {
            TaskType::Classification => DVector::from_vec(vec![0.0; n_points]),
            TaskType::Regression => DVector::from_vec(targets),
        };
        Self {
            inputs: x_matrix,
            targets: y_vector,
            task,
        }
    }
}
