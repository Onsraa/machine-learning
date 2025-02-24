use crate::algorithms::learning_model::LearningModel;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub struct LinearRegression {
    pub weights: DVector<f64>,
    pub bias: f64,
}

impl LinearRegression {
    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = DVector::from_fn(input_dim, |_, _| rng.gen::<f64>() * 0.1);
        let bias = rng.gen::<f64>() * 0.1;
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
    ) -> Vec<f64> {
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
        losses
    }

    pub fn evaluate_scalar(&self, x: &DMatrix<f64>, y: &DVector<f64>) -> f64 {
        let predictions = self.predict_batch(x);
        let errors = &predictions - y;
        errors.map(|e| e * e).mean()
    }
}

impl LearningModel for LinearRegression {
    fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Vec<f64> {
        let y_vec = DVector::from_vec(y.as_slice().to_vec());
        self.fit_scalar(x, &y_vec, learning_rate, n_epochs)
    }

    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
        let y_vec = DVector::from_vec(y.as_slice().to_vec());
        self.evaluate_scalar(x, &y_vec)
    }

    fn predict(&self, x: &DVector<f64>) -> DVector<f64> {
        let x_mat = DMatrix::from_row_slice(1, x.len(), x.as_slice());
        DVector::from_element(1, self.predict_batch(&x_mat)[0])
    }
}
