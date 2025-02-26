use bevy::prelude::*;
use crate::algorithms::mlp::Activation;

#[derive(Resource)]
pub struct MLPConfig {
    pub hidden_layers: Vec<usize>,
    pub hidden_activations: Vec<Activation>,
    pub output_activation: Activation,
    pub dropout_rate: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![5],
            hidden_activations: vec![Activation::Tanh],
            output_activation: Activation::Linear,
            dropout_rate: 0.0,
        }
    }
}

impl MLPConfig {
    /// Creates a configuration suitable for classification
    pub fn for_classification(n_classes: usize) -> Self {
        Self {
            hidden_layers: vec![n_classes * 2],
            hidden_activations: vec![Activation::Tanh],
            output_activation: Activation::Tanh,
            dropout_rate: 0.0,
        }
    }

    /// Creates a configuration suitable for regression
    pub fn for_regression() -> Self {
        Self {
            hidden_layers: vec![10],
            hidden_activations: vec![Activation::Tanh],
            output_activation: Activation::Linear,
            dropout_rate: 0.0,
        }
    }

    /// Gets all activations for creating an MLP
    pub fn get_all_activations(&self) -> Vec<Activation> {
        // Ensure we have enough activations for all hidden layers
        let mut activations = Vec::with_capacity(self.hidden_layers.len() + 1);

        // Add hidden layer activations, recycling if needed
        for i in 0..self.hidden_layers.len() {
            activations.push(
                self.hidden_activations.get(i % self.hidden_activations.len())
                    .copied()
                    .unwrap_or(Activation::Tanh)
            );
        }

        // Add output activation
        activations.push(self.output_activation);

        activations
    }
}