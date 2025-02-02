use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Structure représentant le modèle de régression linéaire
pub struct LinearRegression {
    pub weights: DVector<f64>,
    pub bias: f64,
}

impl LinearRegression {
    /// Crée un nouveau modèle linéaire avec des poids aléatoires
    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = DVector::from_fn(input_dim, |_, _| rng.gen::<f64>() * 0.1);
        let bias = rng.gen::<f64>() * 0.1;

        LinearRegression { weights, bias }
    }

    /// Effectue une prédiction pour un vecteur d'entrée
    pub fn predict(&self, x: &DVector<f64>) -> f64 {
        self.weights.dot(x) + self.bias
    }

    /// Calcule la prédiction pour un lot de données
    pub fn predict_batch(&self, x: &DMatrix<f64>) -> DVector<f64> {
        x * &self.weights + DVector::from_element(x.nrows(), self.bias)
    }

    /// Entraîne le modèle sur les données fournies avec MSE
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(n_epochs);
        let n_samples = x.nrows();

        for _ in 0..n_epochs {
            // Prédictions
            let predictions = self.predict_batch(x);

            // Calcul de l'erreur (MSE)
            let errors = &predictions - y;
            let loss = errors.map(|e| e * e).sum() / (2.0 * n_samples as f64);
            losses.push(loss);

            // Calcul des gradients
            let gradient_weights = (x.transpose() * &errors) / n_samples as f64;
            let gradient_bias = errors.sum() / n_samples as f64;

            // Mise à jour des paramètres
            self.weights -= learning_rate * gradient_weights;
            self.bias -= learning_rate * gradient_bias;
        }

        losses
    }

    /// Calcule le MSE sur un ensemble de données
    pub fn evaluate(&self, x: &DMatrix<f64>, y: &DVector<f64>) -> f64 {
        let predictions = self.predict_batch(x);
        let errors = &predictions - y;
        errors.map(|e| e * e).mean()
    }
}

/// Fonction utilitaire pour convertir des vecteurs en DMatrix/DVector
pub fn prepare_data(x: Vec<Vec<f64>>, y: Vec<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let n_samples = x.len();
    let n_features = x[0].len();

    let x_matrix = DMatrix::from_fn(n_samples, n_features, |i, j| x[i][j]);
    let y_vector = DVector::from_vec(y);

    (x_matrix, y_vector)
}
