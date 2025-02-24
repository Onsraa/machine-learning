use crate::algorithms::learning_model::LearningModel;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub struct LinearClassifier {
    pub n_classes: usize,
    pub classifiers: Vec<DVector<f64>>,
    pub biases: Vec<f64>,
}

impl LinearClassifier {
    pub fn new(input_dim: usize, n_classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let classifiers = (0..n_classes)
            .map(|_| DVector::from_fn(input_dim, |_, _| rng.gen::<f64>() * 0.1))
            .collect();
        let biases = (0..n_classes).map(|_| rng.gen::<f64>() * 0.1).collect();
        Self {
            n_classes,
            classifiers,
            biases,
        }
    }

    fn compute_scores(&self, x: &DVector<f64>) -> Vec<f64> {
        (0..self.n_classes)
            .map(|i| self.classifiers[i].dot(x) + self.biases[i])
            .collect()
    }

    /// Méthode intrinsèque qui retourne un label (usize) pour un vecteur d'entrée.
    pub fn predict_label(&self, x: &DVector<f64>) -> usize {
        // Appliquer tanh pour rendre cohérentes les sorties avec l'entraînement.
        let scores: Vec<f64> = (0..self.n_classes)
            .map(|i| (self.classifiers[i].dot(x) + self.biases[i]).tanh())
            .collect();
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    pub fn predict_batch(&self, x: &DMatrix<f64>) -> Vec<usize> {
        (0..x.nrows())
            .map(|i| {
                let row = DVector::from_iterator(x.ncols(), x.row(i).iter().cloned());
                self.predict_label(&row)
            })
            .collect()
    }

    /// Entraîne le modèle en utilisant une cible sous forme de slice de labels.
    pub fn fit_labels(
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
            for i in 0..x.nrows() {
                let x_i = x.row(i);
                let x_vec = x_i.transpose();
                let mut outputs = Vec::with_capacity(self.n_classes);
                for j in 0..self.n_classes {
                    let activation = self.classifiers[j].dot(&x_vec) + self.biases[j];
                    outputs.push(activation.tanh());
                }
                let true_class = y[i];
                let mut class_loss = 0.0;
                for (j, output) in outputs.iter().enumerate() {
                    let target = if j == true_class { 1.0 } else { -1.0 };
                    let error = output - target;
                    class_loss += error * error;
                }
                total_loss += class_loss;
                for j in 0..self.n_classes {
                    let target = if j == true_class { 1.0 } else { -1.0 };
                    let output = outputs[j];
                    let grad = (output - target) * (1.0 - output * output);
                    self.classifiers[j] -= learning_rate * grad * &x_vec;
                    self.biases[j] -= learning_rate * grad;
                }
            }
            losses.push(total_loss / n_samples);
        }
        losses
    }

    /// Évalue le modèle en calculant l'accuracy et retourne la perte en tant que (1 - accuracy).
    pub fn evaluate_labels(&self, x: &DMatrix<f64>, y: &[usize]) -> f64 {
        let predictions = self.predict_batch(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        1.0 - accuracy // Perte = 1 - accuracy
    }
}

impl LearningModel for LinearClassifier {
    fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Vec<f64> {
        let targets: Vec<usize> = y.column(0).iter().map(|&val| val as usize).collect();
        self.fit_labels(x, &targets, learning_rate, n_epochs)
    }

    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
        let targets: Vec<usize> = y.column(0).iter().map(|&val| val as usize).collect();
        self.evaluate_labels(x, &targets)
    }

    fn predict(&self, x: &DVector<f64>) -> DVector<f64> {
        let class = self.predict_label(x);
        DVector::from_element(1, class as f64)
    }
}
