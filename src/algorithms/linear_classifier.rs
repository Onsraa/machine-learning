use crate::algorithms::learning_model::LearningModel;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::result::Result;

#[derive(Clone)]
pub struct LinearClassifier {
    pub n_classes: usize,
    pub classifiers: Vec<DVector<f64>>,
    pub biases: Vec<f64>,
}

impl LinearClassifier {
    pub fn new(input_dim: usize, n_classes: usize) -> Self {
        if n_classes < 2 {
            panic!("Number of classes must be at least 2");
        }

        let mut rng = rand::thread_rng();
        let classifiers = (0..n_classes)
            .map(|_| DVector::from_fn(input_dim, |_, _| rng.gen_range(-0.1..0.1)))
            .collect();
        let biases = (0..n_classes).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Self {
            n_classes,
            classifiers,
            biases,
        }
    }

    fn compute_scores(&self, x: &DVector<f64>) -> Vec<f64> {
        (0..self.n_classes)
            .map(|i| {
                let value = self.classifiers[i].dot(x) + self.biases[i];
                value.tanh()
            })
            .collect()
    }

    /// Method that returns a label (usize) for an input vector.
    pub fn predict_label(&self, x: &DVector<f64>) -> Result<usize, String> {
        let scores = self.compute_scores(x);
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .ok_or_else(|| "Failed to determine prediction".to_string())
    }

    pub fn predict_batch(&self, x: &DMatrix<f64>) -> Result<Vec<usize>, String> {
        (0..x.nrows())
            .map(|i| {
                let row = DVector::from_iterator(x.ncols(), x.row(i).iter().cloned());
                self.predict_label(&row)
            })
            .collect()
    }

    /// Train the model using a target in the form of a slice of labels.
    pub fn fit_labels(
        &mut self,
        x: &DMatrix<f64>,
        y: &[usize],
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<Vec<f64>, String> {
        // Validate parameters
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if n_epochs == 0 {
            return Err("Number of epochs must be positive".to_string());
        }
        if x.nrows() == 0 || y.is_empty() {
            return Err("Empty dataset".to_string());
        }
        if x.nrows() != y.len() {
            return Err(format!(
                "Number of samples in x ({}) doesn't match y ({})",
                x.nrows(),
                y.len()
            ));
        }

        // Check if any label is >= n_classes
        if let Some(&invalid_label) = y.iter().find(|&&label| label >= self.n_classes) {
            return Err(format!(
                "Invalid label: {}. Maximum allowed is {}",
                invalid_label,
                self.n_classes - 1
            ));
        }

        let mut losses = Vec::with_capacity(n_epochs);
        let n_samples = x.nrows() as f64;

        for _ in 0..n_epochs {
            let mut total_loss = 0.0;
            for i in 0..x.nrows() {
                let x_i = x.row(i);
                let x_vec = x_i.transpose();

                // Forward pass - compute activations
                let mut outputs = Vec::with_capacity(self.n_classes);
                for j in 0..self.n_classes {
                    let activation = self.classifiers[j].dot(&x_vec) + self.biases[j];
                    outputs.push(activation.tanh());
                }

                let true_class = y[i];
                let mut class_loss = 0.0;

                // Compute loss
                for (j, output) in outputs.iter().enumerate() {
                    let target = if j == true_class { 1.0 } else { -1.0 };
                    let error = output - target;
                    class_loss += error * error;
                }
                total_loss += class_loss;

                // Backward pass - update weights
                for j in 0..self.n_classes {
                    let target = if j == true_class { 1.0 } else { -1.0 };
                    let output = outputs[j];

                    // Gradient including tanh derivative (1 - tanh^2)
                    let grad = (output - target) * (1.0 - output * output);
                    self.classifiers[j] -= learning_rate * grad * &x_vec;
                    self.biases[j] -= learning_rate * grad;
                }
            }
            losses.push(total_loss / n_samples);
        }
        Ok(losses)
    }

    /// Evaluate the model by calculating accuracy and returning loss as (1 - accuracy).
    pub fn evaluate_labels(&self, x: &DMatrix<f64>, y: &[usize]) -> Result<f64, String> {
        if x.nrows() == 0 || y.is_empty() {
            return Err("Empty dataset".to_string());
        }
        if x.nrows() != y.len() {
            return Err(format!(
                "Number of samples in x ({}) doesn't match y ({})",
                x.nrows(),
                y.len()
            ));
        }

        // Check if any label is >= n_classes
        if let Some(&invalid_label) = y.iter().find(|&&label| label >= self.n_classes) {
            return Err(format!(
                "Invalid label: {}. Maximum allowed is {}",
                invalid_label,
                self.n_classes - 1
            ));
        }

        let predictions = self.predict_batch(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        Ok(1.0 - accuracy) // Loss = 1 - accuracy
    }
}

impl LearningModel for LinearClassifier {
    fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<Vec<f64>, String> {
        if y.ncols() != 1 {
            return Err(format!("Expected y to have 1 column, got {}", y.ncols()));
        }

        let targets: Result<Vec<usize>, String> = y
            .column(0)
            .iter()
            .map(|&val| {
                if val < 0.0 || val.fract() != 0.0 {
                    Err(format!(
                        "Invalid class label: {}. Must be a non-negative integer",
                        val
                    ))
                } else {
                    Ok(val as usize)
                }
            })
            .collect();

        let targets = targets?;
        self.fit_labels(x, &targets, learning_rate, n_epochs)
    }

    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64, String> {
        if y.ncols() != 1 {
            return Err(format!("Expected y to have 1 column, got {}", y.ncols()));
        }

        let targets: Result<Vec<usize>, String> = y
            .column(0)
            .iter()
            .map(|&val| {
                if val < 0.0 || val.fract() != 0.0 {
                    Err(format!(
                        "Invalid class label: {}. Must be a non-negative integer",
                        val
                    ))
                } else {
                    Ok(val as usize)
                }
            })
            .collect();

        let targets = targets?;
        self.evaluate_labels(x, &targets)
    }

    fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        let class = self.predict_label(x)?;
        Ok(DVector::from_element(1, class as f64))
    }
}
