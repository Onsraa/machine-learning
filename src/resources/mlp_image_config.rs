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
            input_size: 64 * 64,
            output_size: 4,
            hidden_layers: vec![256, 64],
            hidden_activation: Activation::ReLU,
            output_activation: Activation::Tanh,
            learning_rate: 0.001,
            batch_size: 32,
            train_ratio: 0.8,
        }
    }
}