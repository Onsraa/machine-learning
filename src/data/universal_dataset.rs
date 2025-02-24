use nalgebra::{DMatrix, DVector};

/// Indique le type de problème : régression ou classification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Regression,
    Classification,
}

/// Structure universelle pour les données d’entraînement.
pub struct UniversalDataset {
    pub inputs: DMatrix<f64>,
    pub targets: DVector<f64>,
    pub task: TaskType,
}

impl UniversalDataset {
    /// Crée un dataset universel à partir de vecteurs de données.
    pub fn new(inputs: Vec<Vec<f64>>, targets: Vec<f64>, task: TaskType) -> Self {
        let n_samples = inputs.len();
        let n_features = inputs[0].len();
        let x_matrix = DMatrix::from_fn(n_samples, n_features, |i, j| inputs[i][j]);
        let y_vector = DVector::from_vec(targets);
        Self {
            inputs: x_matrix,
            targets: y_vector,
            task,
        }
    }
}
