use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;
use bevy::prelude::*;
use std::collections::VecDeque;

#[derive(Resource)]
pub struct TrainingState {
    pub is_training: bool,
    pub should_reset: bool,
    pub hyperparameters: Hyperparameters,
    pub metrics: TrainingMetrics,
    pub regression_model: Option<LinearRegression>,
    pub classification_model: Option<LinearClassifier>,
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
                epoch_interval: 0.1, // Par défaut, 100ms entre les époques
            },
            metrics: TrainingMetrics::new(1000),
            regression_model: None,
            classification_model: None,
            last_update: 0.0,
        }
    }
}

impl Hyperparameters {
    pub const LEARNING_RATES: [f64; 11] = [
        0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0,
    ];

    pub const TRAIN_RATIOS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    pub const BATCH_SIZES: [usize; 8] = [1, 4, 8, 16, 32, 64, 128, 256];

    pub fn reset_to_defaults(&mut self) {
        *self = Self {
            learning_rate: 0.03,
            train_ratio: 0.9,
            batch_size: 32,
            epoch_interval: 0.1,
        };
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
