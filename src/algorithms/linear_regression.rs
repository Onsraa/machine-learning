use crate::algorithms::learning_model::LearningModel;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::result::Result;

#[derive(Clone)]
pub struct LinearRegression {
    pub weights: DVector<f64>,
    pub bias: f64,
}

impl LinearRegression {
    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (input_dim as f64).sqrt();
        let weights = DVector::from_fn(input_dim, |_, _| rng.gen_range(-scale..scale));
        let bias = rng.gen_range(-0.1..0.1);
        Self { weights, bias }
    }

    pub fn predict_scalar(&self, x: &DVector<f64>) -> f64 {
        self.weights.dot(x) + self.bias
    }

    pub fn predict_batch(&self, x: &DMatrix<f64>) -> DVector<f64> {
        x * &self.weights + DVector::from_element(x.nrows(), self.bias)
    }

    pub fn fit_scalar(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
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
        if x.nrows() == 0 || y.len() == 0 {
            return Err("Empty dataset".to_string());
        }
        if x.nrows() != y.len() {
            return Err(format!(
                "Number of samples in x ({}) doesn't match y ({})",
                x.nrows(),
                y.len()
            ));
        }

        let mut losses = Vec::with_capacity(n_epochs);
        let n_samples = x.nrows();
        for _ in 0..n_epochs {
            let predictions = self.predict_batch(x);
            let errors = &predictions - y;
            let loss = errors.map(|e| e * e).sum() / (2.0 * n_samples as f64);
            losses.push(loss);
            let gradient_weights = (x.transpose() * &errors) / n_samples as f64;
            let gradient_bias = errors.sum() / n_samples as f64;
            self.weights -= learning_rate * gradient_weights;
            self.bias -= learning_rate * gradient_bias;
        }
        Ok(losses)
    }

    pub fn evaluate_scalar(&self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<f64, String> {
        if x.nrows() == 0 || y.len() == 0 {
            return Err("Empty dataset".to_string());
        }
        if x.nrows() != y.len() {
            return Err(format!(
                "Number of samples in x ({}) doesn't match y ({})",
                x.nrows(),
                y.len()
            ));
        }

        let predictions = self.predict_batch(x);
        let errors = &predictions - y;
        Ok(errors.map(|e| e * e).mean())
    }
}

impl LearningModel for LinearRegression {
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
        let y_vec = DVector::from_vec(y.as_slice().to_vec());
        self.fit_scalar(x, &y_vec, learning_rate, n_epochs)
    }

    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64, String> {
        if y.ncols() != 1 {
            return Err(format!("Expected y to have 1 column, got {}", y.ncols()));
        }
        let y_vec = DVector::from_vec(y.as_slice().to_vec());
        self.evaluate_scalar(x, &y_vec)
    }

    fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        let x_mat = DMatrix::from_row_slice(1, x.len(), x.as_slice());
        Ok(DVector::from_element(1, self.predict_batch(&x_mat)[0]))
    }
}