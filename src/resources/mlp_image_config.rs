use crate::algorithms::mlp::Activation;
use bevy::prelude::*;

#[derive(Resource)]
pub struct MLPImageConfig {
    // Architecture
    pub input_size: usize,      // Fixé à 64*64 = 4096
    pub output_size: usize,     // Nombre de classes
    pub hidden_layers: Vec<usize>,

    // Activation
    pub hidden_activation: Activation,
    pub output_activation: Activation,

    // Hyperparamètres
    pub learning_rate: f64,
    pub batch_size: usize,
    pub train_ratio: f64,
    pub max_epochs: usize,

    // Options avancées
    pub use_regularization: bool,
    pub dropout_rate: f64,
    pub early_stopping_patience: usize,
}

impl Default for MLPImageConfig {
    fn default() -> Self {
        Self {
            input_size: 64 * 64,  // Images 64x64
            output_size: 4,       // 4 classes par défaut (FPS, MOBA, RPG, RTS)
            hidden_layers: vec![256, 64],
            hidden_activation: Activation::ReLU,
            output_activation: Activation::Tanh,
            learning_rate: 0.001,
            batch_size: 32,
            train_ratio: 0.8,
            max_epochs: 100,
            use_regularization: false,
            dropout_rate: 0.0,
            early_stopping_patience: 20,
        }
    }
}