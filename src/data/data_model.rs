use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

#[derive(Resource, Default, Debug, Clone, PartialEq, Eq, Hash)]
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

    pub fn is_classification(&self) -> bool {
        matches!(
            self,
            DataModel::LinearSimple
                | DataModel::LinearMultiple
                | DataModel::XOR
                | DataModel::Cross
                | DataModel::MultiLinear3Classes
                | DataModel::MultiCross
        )
    }

    pub fn n_classes(&self) -> Option<usize> {
        if !self.is_classification() {
            return None;
        }
        match self {
            DataModel::MultiLinear3Classes | DataModel::MultiCross => Some(3),
            _ => Some(2),
        }
    }

    pub fn input_dim(&self) -> usize {
        match self {
            DataModel::LinearSimple2d | DataModel::NonLinearSimple2d => 1,
            _ => 2,
        }
    }
}

pub enum DatasetType {
    Regression,
    Classification,
}

pub struct Dataset {
    pub inputs: Vec<Vec<f64>>,
    pub outputs: Vec<f64>,
    pub dataset_type: DatasetType,
    pub n_classes: Option<usize>, // Pour la classification uniquement
}

impl Dataset {
    pub fn to_matrices(&self) -> (DMatrix<f64>, DVector<f64>) {
        let n_samples = self.inputs.len();
        let n_features = self.inputs[0].len();

        let x_matrix = DMatrix::from_fn(n_samples, n_features, |i, j| self.inputs[i][j]);
        let y_vector = DVector::from_vec(self.outputs.clone());

        (x_matrix, y_vector)
    }
}

pub fn generate_dataset(data_model: &DataModel) -> Dataset {
    match data_model {
        // Cas de régression
        DataModel::LinearSimple2d => {
            let inputs = vec![vec![1.0], vec![2.0]];
            let outputs = vec![2.0, 3.0];

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Regression,
                n_classes: None,
            }
        }

        DataModel::LinearSimple3d => {
            let inputs = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 1.0]];
            let outputs = vec![2.0, 3.0, 2.5];

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Regression,
                n_classes: None,
            }
        }

        DataModel::NonLinearSimple2d => {
            let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
            let outputs = vec![2.0, 3.0, 2.5];

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Regression,
                n_classes: None,
            }
        }

        // Cas de classification
        DataModel::LinearMultiple => {
            let mut rng = rand::thread_rng();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();

            // Classe 0
            for _ in 0..50 {
                let x = rng.gen::<f64>() * 0.9 + 1.0;
                let y = rng.gen::<f64>() * 0.9 + 1.0;
                inputs.push(vec![x, y]);
                outputs.push(0.0);
            }

            // Classe 1
            for _ in 0..50 {
                let x = rng.gen::<f64>() * 0.9 + 2.0;
                let y = rng.gen::<f64>() * 0.9 + 2.0;
                inputs.push(vec![x, y]);
                outputs.push(1.0);
            }

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Classification,
                n_classes: Some(2),
            }
        }

        DataModel::XOR => {
            let inputs = vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
            ];
            let outputs = vec![1.0, 1.0, 0.0, 0.0];

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Classification,
                n_classes: Some(2),
            }
        }

        DataModel::Cross => {
            let mut rng = rand::thread_rng();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();

            for _ in 0..500 {
                let x: f64 = rng.gen_range(-1.0..1.0);
                let y: f64 = rng.gen_range(-1.0..1.0);
                let label = if x.abs() <= 0.3 || y.abs() <= 0.3 {
                    1.0
                } else {
                    0.0
                };

                inputs.push(vec![x, y]);
                outputs.push(label);
            }

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Classification,
                n_classes: Some(2),
            }
        }

        DataModel::MultiLinear3Classes => {
            let mut rng = rand::thread_rng();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();

            for _ in 0..500 {
                let x = rng.gen_range(-1.0..1.0);
                let y = rng.gen_range(-1.0..1.0);

                let c1 = -x - y - 0.5 > 0.0 && y < 0.0 && (x - y - 0.5) < 0.0;
                let c2 = -x - y - 0.5 < 0.0 && y > 0.0 && (x - y - 0.5) < 0.0;
                let c3 = -x - y - 0.5 < 0.0 && y < 0.0 && (x - y - 0.5) > 0.0;

                let class = if c1 {
                    0.0
                } else if c2 {
                    1.0
                } else if c3 {
                    2.0
                } else {
                    continue;
                };

                inputs.push(vec![x, y]);
                outputs.push(class);
            }

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Classification,
                n_classes: Some(3),
            }
        }

        DataModel::LinearSimple => {
            // Points colorés : classification binaire
            let inputs = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
            let outputs = vec![0.0, 1.0, 1.0]; // BLUE = 0, RED = 1

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Classification,
                n_classes: Some(2),
            }
        }

        DataModel::LinearTricky3d => {
            let inputs = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
            let outputs = vec![1.0, 2.0, 3.0];

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Regression,
                n_classes: None,
            }
        }

        DataModel::NonLinearSimple3d => {
            let inputs = vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
            ];
            let outputs = vec![2.0, 1.0, -2.0, -1.0];

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Regression,
                n_classes: None,
            }
        }

        DataModel::MultiCross => {
            let mut rng = rand::thread_rng();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();

            for _ in 0..1000 {
                let x = rng.gen_range(-1.0..1.0);
                let y = rng.gen_range(-1.0..1.0);

                let cond_blue: f64 = x % 0.5;
                let cond_red: f64 = y % 0.5;

                let class = if cond_blue.abs() <= 0.25 && cond_red.abs() > 0.25 {
                    0.0 // BLUE
                } else if cond_blue.abs() > 0.25 && cond_red.abs() <= 0.25 {
                    1.0 // RED
                } else {
                    2.0 // GREEN
                };

                inputs.push(vec![x, y]);
                outputs.push(class);
            }

            Dataset {
                inputs,
                outputs,
                dataset_type: DatasetType::Classification,
                n_classes: Some(3),
            }
        }
    }
}
