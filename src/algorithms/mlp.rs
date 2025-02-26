use crate::data::universal_dataset::TaskType;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::result::Result;

#[derive(Clone, Copy)]
pub enum Activation {
    Tanh,
    Linear,
    ReLU,
    Sigmoid,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
            Activation::ReLU => if x > 0.0 { x } else { 0.0 },
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Linear => 1.0,
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            },
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    pub weights: DMatrix<f64>,
    pub biases: DVector<f64>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let scale = (2.0 / (n_in + n_out) as f64).sqrt();

        let weights = DMatrix::from_fn(n_out, n_in, |_, _| {
            rng.gen_range(-scale..scale)
        });

        let biases = DVector::from_fn(n_out, |_, _| {
            rng.gen_range(-0.1..0.1)
        });

        Self {
            weights,
            biases,
            activation,
        }
    }

    pub fn forward(&self, input: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let z = &self.weights * input + &self.biases;
        let a = z.map(|x| self.activation.apply(x));
        (z, a)
    }
}

#[derive(Clone)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    /// Create a new MLP.
    /// The caller must prepare the activation vector according to the problem.
    /// For regression, ensure that the last activation is Linear.
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of the input vectors
    /// * `hidden_layers` - Vector of neuron counts for each hidden layer
    /// * `output_dim` - Dimension of the output (1 for regression, n_classes for classification)
    /// * `activations` - Activation functions for each layer (including output layer)
    ///
    /// # Returns
    /// A new MLP instance
    pub fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        output_dim: usize,
        mut activations: Vec<Activation>,
    ) -> Result<Self, String> {
        if hidden_layers.len() + 1 != activations.len() {
            return Err(format!(
                "Mismatch between layers ({}) and activations ({})",
                hidden_layers.len() + 1,
                activations.len()
            ));
        }

        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for (i, &neurons) in hidden_layers.iter().enumerate() {
            layers.push(Layer::new(prev_dim, neurons, activations[i]));
            prev_dim = neurons;
        }

        layers.push(Layer::new(
            prev_dim,
            output_dim,
            *activations.last().unwrap_or(&Activation::Linear),
        ));

        Ok(Self { layers })
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

    pub fn inference(&self, input: &DVector<f64>) -> Result<DVector<f64>, String> {
        if self.layers.is_empty() {
            return Err("Model has no layers".to_string());
        }

        let forward_result = self.forward(input);
        if let Some(last) = forward_result.last() {
            Ok(last.1.clone())
        } else {
            Err("Forward pass failed".to_string())
        }
    }

    /// Train the MLP on a single example.
    /// The `task` parameter allows adapting the target conversion.
    pub fn train_example(
        &mut self,
        input: &DVector<f64>,
        target: &DVector<f64>,
        learning_rate: f64,
        task: TaskType,
    ) -> Result<f64, String> {
        if self.layers.is_empty() {
            return Err("Model has no layers".to_string());
        }

        let forward_cache = self.forward(input);
        let output = forward_cache.last().unwrap().1.clone();

        let target_adjusted = if task == TaskType::Classification {
            let class = target[0] as usize;
            let mut v = DVector::from_element(output.len(), -1.0);
            if class < output.len() {
                v[class] = 1.0;
            } else {
                return Err(format!("Invalid class label: {}. Maximum class is {}", class, output.len() - 1));
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
            let a_prev = if i == 0 {
                input
            } else {
                &forward_cache[i - 1].1
            };

            let delta_current = if let Some(delta_next) = &delta {
                let next_weights = self.layers[i + 1].weights.clone();
                let delta_temp = next_weights.transpose() * delta_next;
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

        Ok(loss)
    }

    pub fn fit(
        &mut self,
        inputs: &DMatrix<f64>,
        targets: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
        task: TaskType,
    ) -> Result<Vec<f64>, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if n_epochs == 0 {
            return Err("Number of epochs must be positive".to_string());
        }
        if inputs.nrows() == 0 || targets.nrows() == 0 {
            return Err("Empty dataset".to_string());
        }
        if inputs.nrows() != targets.nrows() {
            return Err(format!(
                "Number of samples in inputs ({}) doesn't match targets ({})",
                inputs.nrows(),
                targets.nrows()
            ));
        }

        let n_samples = inputs.nrows();
        let mut losses = Vec::with_capacity(n_epochs);

        for _ in 0..n_epochs {
            let mut total_loss = 0.0;
            for i in 0..n_samples {
                let x_i = DVector::from_iterator(inputs.ncols(), inputs.row(i).iter().cloned());
                let target_i =
                    DVector::from_iterator(targets.ncols(), targets.row(i).iter().cloned());
                total_loss += self.train_example(&x_i, &target_i, learning_rate, task)?;
            }
            losses.push(total_loss / n_samples as f64);
        }

        Ok(losses)
    }

    pub fn evaluate(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>, task: TaskType) -> Result<f64, String> {
        if inputs.nrows() == 0 || targets.nrows() == 0 {
            return Err("Empty dataset".to_string());
        }
        if inputs.nrows() != targets.nrows() {
            return Err(format!(
                "Number of samples in inputs ({}) doesn't match targets ({})",
                inputs.nrows(),
                targets.nrows()
            ));
        }

        let n_samples = inputs.nrows();
        let mut total_loss = 0.0;

        for i in 0..n_samples {
            let x_i = DVector::from_iterator(inputs.ncols(), inputs.row(i).iter().cloned());
            let target_i = DVector::from_iterator(targets.ncols(), targets.row(i).iter().cloned());
            let output = self.inference(&x_i)?;

            let target_adjusted = if task == TaskType::Classification {
                let class = target_i[0] as usize;
                let mut v = DVector::from_element(output.len(), -1.0);
                if class < output.len() {
                    v[class] = 1.0;
                } else {
                    return Err(format!("Invalid class label: {}. Maximum class is {}", class, output.len() - 1));
                }
                v
            } else {
                target_i.clone()
            };

            let error = output - target_adjusted;
            total_loss += error.dot(&error) / 2.0;
        }

        Ok(total_loss / n_samples as f64)
    }
}

use crate::algorithms::learning_model::LearningModel;

impl LearningModel for MLP {
    fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<Vec<f64>, String> {
        // By default, assume classification
        self.fit(
            x,
            y,
            learning_rate,
            n_epochs,
            TaskType::Classification,
        )
    }

    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64, String> {
        self.evaluate(
            x,
            y,
            TaskType::Classification,
        )
    }

    fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        self.inference(x)
    }
}