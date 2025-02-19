use bevy::prelude::*;

#[derive(Resource)]
pub struct MLPConfig {
    pub hidden_layers: Vec<usize>,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![5],
        }
    }
}
