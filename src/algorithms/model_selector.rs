// src/algorithms/model_selector.rs

use nalgebra::{DMatrix, DVector};
use crate::algorithms::linear_regression::LinearRegression;
use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::mlp::MLP;

/// Enumération pour encapsuler les modèles disponibles.
pub enum ModelAlgorithm {
    LinearRegression(LinearRegression),
    LinearClassifier(LinearClassifier),
    MLP(MLP),
    // Extensions futures : SVM, RBF, etc.
}

impl ModelAlgorithm {
    pub fn fit(&mut self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>, learning_rate: f64, n_epochs: usize) -> f64 {
        match self {
            ModelAlgorithm::LinearRegression(model) => {
                let targets_vec = nalgebra::DVector::from_vec(targets.as_slice().to_vec());
                let losses = model.fit(inputs, &targets_vec, learning_rate, n_epochs);
                losses.last().cloned().unwrap_or(0.0)
            },
            ModelAlgorithm::LinearClassifier(model) => {
                let targets_vec: Vec<usize> = targets.iter().map(|&x| x as usize).collect();
                let losses = model.fit(inputs, &targets_vec, learning_rate, n_epochs);
                losses.last().cloned().unwrap_or(0.0)
            },
            ModelAlgorithm::MLP(model) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs);
                losses.last().cloned().unwrap_or(0.0)
            },
        }
    }

    pub fn evaluate(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> f64 {
        match self {
            ModelAlgorithm::LinearRegression(model) => {
                let targets_vec = nalgebra::DVector::from_vec(targets.as_slice().to_vec());
                model.evaluate(inputs, &targets_vec)
            },
            ModelAlgorithm::LinearClassifier(model) => {
                let targets_vec: Vec<usize> = targets.iter().map(|&x| x as usize).collect();
                1.0 - model.evaluate(inputs, &targets_vec)
            },
            ModelAlgorithm::MLP(model) => {
                model.evaluate(inputs, targets)
            },
        }
    }
}
