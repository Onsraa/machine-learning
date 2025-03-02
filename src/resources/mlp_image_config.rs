use crate::algorithms::mlp::Activation;
use bevy::prelude::*;

#[derive(Resource)]
pub struct MLPImageConfig {
    // Architecture
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layers: Vec<usize>,

    // Activation
    pub hidden_activation: Activation,
    pub output_activation: Activation,

    // Hyperparamètres
    pub learning_rate: f64,
    pub batch_size: usize,
    pub train_ratio: f64,
}

impl Default for MLPImageConfig {
    fn default() -> Self {
        Self {
            input_size: 128 * 72 * 3,
            output_size: 4,
            hidden_layers: vec![2048, 512],
            hidden_activation: Activation::ReLU,
            output_activation: Activation::Tanh,
            learning_rate: 0.0005,
            batch_size: 32,
            train_ratio: 0.8,
        }
    }
}