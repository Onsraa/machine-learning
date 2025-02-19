// src/algorithms/mlp.rs

use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Enumération pour les fonctions d'activation.
#[derive(Clone, Copy)]
pub enum Activation {
    Tanh,
    // D'autres fonctions pourront être ajoutées ultérieurement.
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - x.tanh().powi(2),
        }
    }
}

/// Structure représentant une couche du MLP.
pub struct Layer {
    pub weights: DMatrix<f64>,
    pub biases: DVector<f64>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let weights = DMatrix::from_fn(n_out, n_in, |_, _| rng.gen::<f64>() * 0.1);
        let biases = DVector::from_fn(n_out, |_, _| rng.gen::<f64>() * 0.1);
        Self { weights, biases, activation }
    }

    pub fn forward(&self, input: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let z = &self.weights * input + &self.biases;
        let a = z.map(|x| self.activation.apply(x));
        (z, a)
    }
}

/// Structure représentant l'ensemble du MLP.
pub struct MLP {
    pub layers: Vec<Layer>,
    pub is_classification: bool,
}

impl MLP {
    pub fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        output_dim: usize,
        activations: Vec<Activation>,
        is_classification: bool,
    ) -> Self {
        assert_eq!(hidden_layers.len() + 1, activations.len(), "Mismatch between layers and activations");
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;
        for (i, &neurons) in hidden_layers.iter().enumerate() {
            layers.push(Layer::new(prev_dim, neurons, activations[i]));
            prev_dim = neurons;
        }
        layers.push(Layer::new(prev_dim, output_dim, *activations.last().unwrap()));
        Self { layers, is_classification }
    }

    pub fn forward(&self, input: &DVector<f64>) -> Vec<(DVector<f64>, DVector<f64>)> {
        let mut cache = Vec::new();
        let mut a = input.clone();
        for layer in &self.layers {
            let (z, a_next) = layer.forward(&a);
            cache.push((z, a_next.clone()));
            a = a_next;
        }
        cache
    }

    pub fn predict(&self, input: &DVector<f64>) -> DVector<f64> {
        self.forward(input).last().unwrap().1.clone()
    }

    /// Entraîne le MLP sur un seul exemple.
    /// Pour la classification, la cible (une valeur scalaire dans un DVector de taille 1)
    /// est convertie en vecteur one‑hot (1 pour la classe correcte, -1 pour les autres).
    pub fn train_example(&mut self, input: &DVector<f64>, target: &DVector<f64>, learning_rate: f64) -> f64 {
        let forward_cache = self.forward(input);
        let output = forward_cache.last().unwrap().1.clone();

        let target_adjusted = if self.is_classification {
            let class = target[0] as usize;
            let mut v = DVector::from_element(output.len(), -1.0);
            if class < output.len() {
                v[class] = 1.0;
            }
            v
        } else {
            target.clone()
        };

        let error = &output - &target_adjusted;
        let loss = error.dot(&error) / 2.0;

        let mut delta: Option<DVector<f64>> = None;
        for i in (0..self.layers.len()).rev() {
            let (z, a) = &forward_cache[i];
            let a_prev = if i == 0 { input } else { &forward_cache[i - 1].1 };

            let delta_current = if let Some(delta_next) = &delta {
                // Pour les couches cachées, cloner les poids de la couche suivante.
                let next_weights = if i < self.layers.len() - 1 {
                    self.layers[i + 1].weights.clone()
                } else {
                    DMatrix::zeros(0, 0)
                };
                let delta_temp = next_weights.transpose() * delta_next;
                // Copie de l'activation pour éviter de réemprunter self.layers[i] mutuellement.
                let act = self.layers[i].activation;
                DVector::from_fn(a.len(), |j, _| {
                    let grad = act.derivative(z[j]);
                    delta_temp[j] * grad
                })
            } else {
                let act = self.layers[i].activation;
                DVector::from_fn(a.len(), |j, _| {
                    let grad = act.derivative(z[j]);
                    (a[j] - target_adjusted[j]) * grad
                })
            };

            {
                let current_layer = &mut self.layers[i];
                let grad_w = &delta_current * a_prev.transpose();
                current_layer.weights -= learning_rate * grad_w;
                current_layer.biases -= learning_rate * delta_current.clone();
            }
            delta = Some(delta_current);
        }

        loss
    }

    pub fn fit(&mut self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>, learning_rate: f64, n_epochs: usize) -> Vec<f64> {
        let n_samples = inputs.nrows();
        let mut losses = Vec::with_capacity(n_epochs);
        for _ in 0..n_epochs {
            let mut total_loss = 0.0;
            for i in 0..n_samples {
                let x_i = DVector::from_iterator(inputs.ncols(), inputs.row(i).iter().cloned());
                let target_i = DVector::from_iterator(targets.ncols(), targets.row(i).iter().cloned());
                total_loss += self.train_example(&x_i, &target_i, learning_rate);
            }
            losses.push(total_loss / n_samples as f64);
        }
        losses
    }

    pub fn evaluate(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> f64 {
        let n_samples = inputs.nrows();
        let mut total_loss = 0.0;
        for i in 0..n_samples {
            let x_i = DVector::from_iterator(inputs.ncols(), inputs.row(i).iter().cloned());
            let target_i = DVector::from_iterator(targets.ncols(), targets.row(i).iter().cloned());
            let output = self.predict(&x_i);
            let target_adjusted = if self.is_classification {
                let class = target_i[0] as usize;
                let mut v = DVector::from_element(output.len(), -1.0);
                if class < output.len() {
                    v[class] = 1.0;
                }
                v
            } else {
                target_i.clone()
            };
            let error = output - target_adjusted;
            total_loss += error.dot(&error) / 2.0;
        }
        total_loss / n_samples as f64
    }
}
