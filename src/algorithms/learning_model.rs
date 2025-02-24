use nalgebra::{DMatrix, DVector};

/// Trait commun pour tous les modèles d’apprentissage.
/// Chaque modèle doit implémenter fit(), evaluate() et predict().
pub trait LearningModel {
    /// Entraîne le modèle sur les données d’entrée `x` et les cibles `y`
    /// pendant `n_epochs` avec le taux d’apprentissage `learning_rate`.
    /// Retourne un vecteur contenant la perte moyenne par époque.
    fn fit(&mut self, x: &DMatrix<f64>, y: &DMatrix<f64>, learning_rate: f64, n_epochs: usize) -> Vec<f64>;

    /// Évalue le modèle sur un jeu de données et retourne une mesure d’erreur.
    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> f64;

    /// Effectue une prédiction pour une entrée donnée.
    fn predict(&self, x: &DVector<f64>) -> DVector<f64>;
}
