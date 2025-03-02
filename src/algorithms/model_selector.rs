use crate::algorithms::learning_model::LearningModel;
use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;
use crate::algorithms::mlp::MLP;
use crate::algorithms::rbf::RBF;
use crate::algorithms::svm::{SVM};
use crate::data::universal_dataset::TaskType;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::result::Result;

/// Enum that wraps all model algorithms and tracks their task type
#[derive(Clone, Serialize, Deserialize)]
pub enum ModelAlgorithm {
    LinearRegression(LinearRegression, TaskType),
    LinearClassifier(LinearClassifier, TaskType),
    MLP(MLP, TaskType),
    RBF(RBF, TaskType),
    SVM(SVM, TaskType),
}

impl ModelAlgorithm {
    /// Creates a new LinearRegression model
    pub fn new_linear_regression(input_dim: usize) -> Self {
        ModelAlgorithm::LinearRegression(LinearRegression::new(input_dim), TaskType::Regression)
    }

    /// Creates a new LinearClassifier model
    pub fn new_linear_classifier(input_dim: usize, n_classes: usize) -> Self {
        ModelAlgorithm::LinearClassifier(
            LinearClassifier::new(input_dim, n_classes),
            TaskType::Classification,
        )
    }

    /// Creates a new MLP model
    pub fn new_mlp(mlp: MLP, is_classification: bool) -> Self {
        let task_type = if is_classification {
            TaskType::Classification
        } else {
            TaskType::Regression
        };

        ModelAlgorithm::MLP(mlp, task_type)
    }

    /// Creates a new RBF model
    pub fn new_rbf(rbf: RBF) -> Self {
        let task_type = if rbf.is_classification {
            TaskType::Classification
        } else {
            TaskType::Regression
        };

        ModelAlgorithm::RBF(rbf, task_type)
    }

    /// Creates a new SVM model
    pub fn new_svm(svm: SVM) -> Self {
        // SVM est toujours pour la classification binaire
        ModelAlgorithm::SVM(svm, TaskType::Classification)
    }

    /// Returns whether this model is for classification or regression
    pub fn is_classification(&self) -> bool {
        match self {
            ModelAlgorithm::LinearRegression(_, _) => false,
            ModelAlgorithm::LinearClassifier(_, _) => true,
            ModelAlgorithm::MLP(_, task_type) => *task_type == TaskType::Classification,
            ModelAlgorithm::RBF(rbf, _) => rbf.is_classification,
            ModelAlgorithm::SVM(_, _) => true,
        }
    }

    /// Gets the task type of this model
    pub fn get_task_type(&self) -> TaskType {
        match self {
            ModelAlgorithm::LinearRegression(_, task_type) => *task_type,
            ModelAlgorithm::LinearClassifier(_, task_type) => *task_type,
            ModelAlgorithm::MLP(_, task_type) => *task_type,
            ModelAlgorithm::RBF(_, task_type) => *task_type,
            ModelAlgorithm::SVM(_, task_type) => *task_type,
        }
    }

    pub fn fit(
        &mut self,
        inputs: &DMatrix<f64>,
        targets: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<f64, String> {
        match self {
            ModelAlgorithm::LinearRegression(model, _) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs)?;
                Ok(*losses.last().unwrap_or(&0.0))
            }
            ModelAlgorithm::LinearClassifier(model, _) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs)?;
                Ok(*losses.last().unwrap_or(&0.0))
            }
            ModelAlgorithm::MLP(model, task_type) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs, *task_type)?;
                Ok(*losses.last().unwrap_or(&0.0))
            }
            ModelAlgorithm::RBF(model, _) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs)?;
                Ok(*losses.last().unwrap_or(&0.0))
            }
            ModelAlgorithm::SVM(model, _) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs)?;
                Ok(*losses.last().unwrap_or(&0.0))
            }
        }
    }

    pub fn evaluate(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> Result<f64, String> {
        match self {
            ModelAlgorithm::LinearRegression(model, _) => model.evaluate(inputs, targets),
            ModelAlgorithm::LinearClassifier(model, _) => model.evaluate(inputs, targets),
            ModelAlgorithm::MLP(model, task_type) => model.evaluate(inputs, targets, *task_type),
            ModelAlgorithm::RBF(model, _) => model.evaluate(inputs, targets),
            ModelAlgorithm::SVM(model, _) => model.evaluate(inputs, targets),
        }
    }

    pub fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        match self {
            ModelAlgorithm::LinearRegression(model, _) => model.predict(x),
            ModelAlgorithm::LinearClassifier(model, _) => model.predict(x),
            ModelAlgorithm::MLP(model, _) => model.predict(x),
            ModelAlgorithm::RBF(model, _) => model.predict(x),
            ModelAlgorithm::SVM(model, _) => model.predict(x),
        }
    }
}
