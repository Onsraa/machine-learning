use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Structure représentant le modèle de classification linéaire multi-classes
pub struct LinearClassifier {
    pub n_classes: usize,
    pub classifiers: Vec<DVector<f64>>,
    pub biases: Vec<f64>,
}

impl LinearClassifier {
    /// Crée un nouveau modèle de classification avec des poids aléatoires
    pub fn new(input_dim: usize, n_classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let classifiers = (0..n_classes)
            .map(|_| DVector::from_fn(input_dim, |_, _| rng.gen::<f64>() * 0.1))
            .collect();
        let biases = (0..n_classes).map(|_| rng.gen::<f64>() * 0.1).collect();

        LinearClassifier {
            n_classes,
            classifiers,
            biases,
        }
    }

    /// Calcule les scores pour chaque classe
    fn compute_scores(&self, x: &DVector<f64>) -> Vec<f64> {
        (0..self.n_classes)
            .map(|i| self.classifiers[i].dot(x) + self.biases[i])
            .collect()
    }

    /// Prédit la classe avec le score le plus élevé
    pub fn predict(&self, x: &DVector<f64>) -> usize {
        let scores = self.compute_scores(x);
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    /// Prédit pour un lot de données
    pub fn predict_batch(&self, x: &DMatrix<f64>) -> Vec<usize> {
        (0..x.nrows())
            .map(|i| {
                let row = DVector::from_iterator(x.ncols(), x.row(i).iter().cloned());
                self.predict(&row)
            })
            .collect()
    }

    /// Convertit les labels en format one-hot
    fn to_one_hot(&self, y: &[usize]) -> DMatrix<f64> {
        let n_samples = y.len();
        let mut one_hot = DMatrix::zeros(n_samples, self.n_classes);

        for (i, &label) in y.iter().enumerate() {
            one_hot[(i, label)] = 1.0;
        }

        one_hot
    }

    /// Entraîne le modèle avec softmax et cross-entropy
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &[usize],
        learning_rate: f64,
        n_epochs: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(n_epochs);
        let n_samples = x.nrows() as f64;

        for _ in 0..n_epochs {
            let mut total_loss = 0.0;

            // Pour chaque exemple
            for i in 0..x.nrows() {
                let x_i = x.row(i);
                let x_vec = x_i.transpose();

                // Calcul des scores et softmax
                let mut scores = self.compute_scores(&x_vec);
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                }
                let sum_exp = scores.iter().sum::<f64>();
                for score in &mut scores {
                    *score /= sum_exp;
                }

                // Cross-entropy loss
                let true_class = y[i];
                total_loss -= (scores[true_class] + 1e-10).ln();

                // Gradients et mise à jour
                for j in 0..self.n_classes {
                    let target = if j == true_class { 1.0 } else { 0.0 };
                    let error = scores[j] - target;

                    self.classifiers[j] -= &(learning_rate * error * &x_vec / n_samples);
                    self.biases[j] -= learning_rate * error / n_samples;
                }
            }

            losses.push(total_loss / n_samples);
        }

        losses
    }

    /// Évalue l'accuracy du modèle
    pub fn evaluate(&self, x: &DMatrix<f64>, y: &[usize]) -> f64 {
        let predictions = self.predict_batch(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_)| pred == true_)
            .count();

        correct as f64 / y.len() as f64
    }
}
