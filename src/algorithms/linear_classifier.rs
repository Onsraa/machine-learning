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

    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    pub fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], learning_rate: f64, n_epochs: usize) -> Vec<f64> {
        let mut losses = Vec::with_capacity(n_epochs);
        let n_samples = x.nrows() as f64;

        for _ in 0..n_epochs {
            let mut total_loss = 0.0;

            for i in 0..x.nrows() {
                let x_i = x.row(i);
                let x_vec = x_i.transpose();

                // Calcul des sorties avec tanh
                let mut outputs = Vec::with_capacity(self.n_classes);
                for j in 0..self.n_classes {
                    let activation = self.classifiers[j].dot(&x_vec) + self.biases[j];
                    outputs.push(Self::tanh(activation));
                }

                let true_class = y[i];

                // Calcul de la perte (MSE au lieu de cross-entropy)
                let mut class_loss = 0.0;
                for (j, output) in outputs.iter().enumerate() {
                    let target = if j == true_class { 1.0 } else { -1.0 };
                    let error = output - target;
                    class_loss += error * error;
                }
                total_loss += class_loss;

                // Mise à jour avec la dérivée de tanh
                for j in 0..self.n_classes {
                    let target = if j == true_class { 1.0 } else { -1.0 };
                    let output = outputs[j];
                    // dérivée de tanh = 1 - tanh²(x)
                    let grad = (output - target) * (1.0 - output * output);

                    self.classifiers[j] -= learning_rate * grad * &x_vec;
                    self.biases[j] -= learning_rate * grad;
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
