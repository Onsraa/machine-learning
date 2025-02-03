use bevy::prelude::*;
use std::collections::VecDeque;
use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;

#[derive(Resource)]
pub struct TrainingState {
    pub is_training: bool,
    pub should_reset: bool,  // Nouveau flag pour le reset
    pub hyperparameters: Hyperparameters,
    pub metrics: TrainingMetrics,
    pub regression_model: Option<LinearRegression>,
    pub classification_model: Option<LinearClassifier>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            is_training: false,
            should_reset: false,
            hyperparameters: Hyperparameters::default(),
            metrics: TrainingMetrics::new(1000),
            regression_model: None,
            classification_model: None,
        }
    }
}

#[derive(Default)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub train_ratio: f64,
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