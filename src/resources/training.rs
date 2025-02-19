use crate::algorithms::model_selector::ModelAlgorithm;
use bevy::prelude::*;
use std::collections::VecDeque;

#[derive(Resource)]
pub struct TrainingState {
    pub is_training: bool,
    pub should_reset: bool,
    pub hyperparameters: Hyperparameters,
    pub metrics: TrainingMetrics,
    pub selected_model: Option<ModelAlgorithm>,
    pub last_update: f32,
}

#[derive(Default)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub train_ratio: f64,
    pub batch_size: usize,
    pub epoch_interval: f32,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            is_training: false,
            should_reset: false,
            hyperparameters: Hyperparameters {
                learning_rate: 0.03,
                train_ratio: 0.9,
                batch_size: 32,
                epoch_interval: 0.1,
            },
            metrics: TrainingMetrics::new(1000),
            selected_model: None,
            last_update: 0.0,
        }
    }
}

#[derive(Default)]
pub struct TrainingMetrics {
    pub training_losses: VecDeque<f64>,
    pub test_losses: VecDeque<f64>,
    pub current_epoch: usize,
    pub max_history: usize,
}

impl TrainingMetrics {
    pub fn new(max_history: usize) -> Self {
        Self {
            training_losses: VecDeque::with_capacity(max_history),
            test_losses: VecDeque::with_capacity(max_history),
            current_epoch: 0,
            max_history,
        }
    }

    pub fn add_metrics(&mut self, train_loss: f64, test_loss: f64) {
        if self.training_losses.len() >= self.max_history {
            self.training_losses.pop_front();
            self.test_losses.pop_front();
        }
        self.training_losses.push_back(train_loss);
        self.test_losses.push_back(test_loss);
        self.current_epoch += 1;
    }

    pub fn reset(&mut self) {
        self.training_losses.clear();
        self.test_losses.clear();
        self.current_epoch = 0;
    }
}
