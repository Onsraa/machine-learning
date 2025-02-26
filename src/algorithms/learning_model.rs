use nalgebra::{DMatrix, DVector};
use std::result::Result;

/// Common trait for all learning models.
/// Each model must implement fit(), evaluate() and predict().
pub trait LearningModel {
    /// Train the model on input data `x` and target `y`
    /// for `n_epochs` with learning rate `learning_rate`.
    /// Returns a vector containing the average loss per epoch.
    fn fit(&mut self, x: &DMatrix<f64>, y: &DMatrix<f64>, learning_rate: f64, n_epochs: usize) -> Result<Vec<f64>, String>;

    /// Evaluate the model on a dataset and return an error measure.
    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64, String>;

    /// Make a prediction for a given input.
    fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String>;
}