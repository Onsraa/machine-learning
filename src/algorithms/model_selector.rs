use crate::algorithms::learning_model::LearningModel;
use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;
use crate::algorithms::mlp::MLP;
use nalgebra::{DMatrix, DVector};

/// Enumération pour encapsuler les modèles d’apprentissage disponibles.
pub enum ModelAlgorithm {
    LinearRegression(LinearRegression),
    LinearClassifier(LinearClassifier),
    MLP(MLP),
    // Extensions futures : SVM, RBF, etc.
}

impl ModelAlgorithm {
    /// Entraîne le modèle sur les données d’entrée `inputs` et les cibles `targets`
    /// pendant `n_epochs` avec le taux d’apprentissage `learning_rate`.
    /// Retourne la dernière perte moyenne calculée.
    pub fn fit(
        &mut self,
        inputs: &DMatrix<f64>,
        targets: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> f64 {
        match self {
            ModelAlgorithm::LinearRegression(model) => {
                let losses = model.fit(inputs, targets, learning_rate, n_epochs);
                *losses.last().unwrap_or(&0.0)
            }
            ModelAlgorithm::LinearClassifier(model) => {
                let losses = <LinearClassifier as LearningModel>::fit(
                    model,
                    inputs,
                    targets,
                    learning_rate,
                    n_epochs,
                );
                *losses.last().unwrap_or(&0.0)
            }
            ModelAlgorithm::MLP(model) => {
                let losses =
                    <MLP as LearningModel>::fit(model, inputs, targets, learning_rate, n_epochs);
                *losses.last().unwrap_or(&0.0)
            }
        }
    }

    /// Évalue le modèle sur les données et retourne une mesure d’erreur.
    pub fn evaluate(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> f64 {
        match self {
            ModelAlgorithm::LinearRegression(model) => model.evaluate(inputs, targets),
            ModelAlgorithm::LinearClassifier(model) => {
                <LinearClassifier as LearningModel>::evaluate(model, inputs, targets)
            }
            ModelAlgorithm::MLP(model) => <MLP as LearningModel>::evaluate(model, inputs, targets),
        }
    }

    /// Effectue une prédiction pour une entrée donnée.
    pub fn predict(&self, x: &DVector<f64>) -> DVector<f64> {
        match self {
            ModelAlgorithm::LinearRegression(model) => model.predict(x),
            ModelAlgorithm::LinearClassifier(model) => {
                <LinearClassifier as LearningModel>::predict(model, x)
            }
            ModelAlgorithm::MLP(model) => <MLP as LearningModel>::predict(model, x),
        }
    }
}
