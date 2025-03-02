use crate::algorithms::mlp::Activation;
use bevy::prelude::*;

#[derive(Resource)]
pub struct MLPConfig {
    pub hidden_layers: Vec<usize>,
    pub hidden_activations: Vec<Activation>,
    pub output_activation: Activation,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![5],
            hidden_activations: vec![Activation::Tanh],
            output_activation: Activation::Linear,
        }
    }
}