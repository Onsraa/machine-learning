use nalgebra::{DMatrix, DVector};
use std::result::Result;
use serde::{Serialize, Deserialize};

/// Indicates the problem type: regression or classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TaskType {
    Regression,
    Classification,
}

/// Universal structure for training data.
pub struct UniversalDataset {
    pub inputs: DMatrix<f64>,
    pub targets: DVector<f64>,
    pub task: TaskType,
    pub n_classes: Option<usize>, // Only for classification
}

impl Default for UniversalDataset {
    fn default() -> Self {
        Self {
            inputs: DMatrix::zeros(0, 0),
            targets: DVector::zeros(0),
            task: TaskType::Classification,
            n_classes: None,
        }
    }
}

impl UniversalDataset {
    /// Creates a universal dataset from data vectors.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input feature vectors
    /// * `targets` - Vector of target values
    /// * `task` - Task type (Regression or Classification)
    /// * `n_classes` - Number of classes (only for Classification tasks)
    ///
    /// # Returns
    /// A Result containing the dataset or an error message
    pub fn new(
        inputs: Vec<Vec<f64>>,
        targets: Vec<f64>,
        task: TaskType,
        n_classes: Option<usize>
    ) -> Result<Self, String> {
        if inputs.is_empty() || targets.is_empty() {
            return Err("Empty dataset".to_string());
        }

        if inputs.len() != targets.len() {
            return Err(format!(
                "Number of input samples ({}) doesn't match target samples ({})",
                inputs.len(),
                targets.len()
            ));
        }

        let n_features = inputs[0].len();
        for (i, input) in inputs.iter().enumerate() {
            if input.len() != n_features {
                return Err(format!(
                    "Inconsistent input dimensions: sample {} has {} features, expected {}",
                    i, input.len(), n_features
                ));
            }
        }

        if task == TaskType::Classification {
            if n_classes.is_none() {
                return Err("Number of classes must be specified for classification tasks".to_string());
            }

            let n_classes = n_classes.unwrap();
            if n_classes < 2 {
                return Err("Classification tasks must have at least 2 classes".to_string());
            }

            for (i, &target) in targets.iter().enumerate() {
                if target < 0.0 || target.fract() != 0.0 || target as usize >= n_classes {
                    return Err(format!(
                        "Invalid class label at index {}: {}. Must be an integer in range [0, {}]",
                        i, target, n_classes - 1
                    ));
                }
            }
        }

        let n_samples = inputs.len();
        let x_matrix = DMatrix::from_fn(n_samples, n_features, |i, j| inputs[i][j]);
        let y_vector = DVector::from_vec(targets);

        Ok(Self {
            inputs: x_matrix,
            targets: y_vector,
            task,
            n_classes,
        })
    }

    /// Splits dataset into training and testing sets
    ///
    /// # Arguments
    /// * `train_ratio` - Ratio of data to use for training (0.0 to 1.0)
    ///
    /// # Returns
    /// A tuple of (training dataset, testing dataset)
    pub fn split(&self, train_ratio: f64) -> Result<(Self, Self), String> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(format!("Invalid train_ratio: {}. Must be between 0.0 and 1.0", train_ratio));
        }

        let n_samples = self.inputs.nrows();
        let n_train = (n_samples as f64 * train_ratio) as usize;

        if n_train == 0 || n_train == n_samples {
            return Err(format!(
                "Train/test split resulted in empty set. Adjust train_ratio ({}) or use more data",
                train_ratio
            ));
        }

        // Create training set
        let train_inputs = self.inputs.rows(0, n_train).clone_owned();
        let train_targets = self.targets.rows(0, n_train).clone_owned();

        // Create testing set
        let test_inputs = self.inputs.rows(n_train, n_samples - n_train).clone_owned();
        let test_targets = self.targets.rows(n_train, n_samples - n_train).clone_owned();

        let train_set = Self {
            inputs: train_inputs,
            targets: train_targets,
            task: self.task,
            n_classes: self.n_classes,
        };

        let test_set = Self {
            inputs: test_inputs,
            targets: test_targets,
            task: self.task,
            n_classes: self.n_classes,
        };

        Ok((train_set, test_set))
    }

    /// Normalizes the dataset features (standardization)
    ///
    /// # Returns
    /// A new dataset with normalized features
    pub fn normalize(&self) -> Self {
        let n_features = self.inputs.ncols();
        let mut normalized_inputs = self.inputs.clone();

        for j in 0..n_features {
            let column = self.inputs.column(j);
            let mean = column.mean();
            let std_dev = (column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64).sqrt();

            if std_dev > 1e-10 {
                for i in 0..self.inputs.nrows() {
                    normalized_inputs[(i, j)] = (self.inputs[(i, j)] - mean) / std_dev;
                }
            }
        }

        Self {
            inputs: normalized_inputs,
            targets: self.targets.clone(),
            task: self.task,
            n_classes: self.n_classes,
        }
    }
}