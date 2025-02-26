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
    pub error_message: Option<String>,
}

pub struct Hyperparameters {
    pub learning_rate: f64,
    pub train_ratio: f64,
    pub batch_size: usize,
    pub epoch_interval: f32,
    pub early_stopping_patience: usize,
    pub regularization_strength: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            train_ratio: 0.8,
            batch_size: 32,
            epoch_interval: 0.1,
            early_stopping_patience: 300,
            regularization_strength: 0.0,
        }
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            is_training: false,
            should_reset: false,
            hyperparameters: Hyperparameters::default(),
            metrics: TrainingMetrics::new(1000),
            selected_model: None,
            last_update: 0.0,
            error_message: None,
        }
    }
}

#[derive(Default)]
pub struct TrainingMetrics {
    pub training_losses: VecDeque<f64>,
    pub test_losses: VecDeque<f64>,
    pub current_epoch: usize,
    pub max_history: usize,
    pub best_test_loss: f64,
    pub epochs_since_improvement: usize,
}

impl TrainingMetrics {
    pub fn new(max_history: usize) -> Self {
        Self {
            training_losses: VecDeque::with_capacity(max_history),
            test_losses: VecDeque::with_capacity(max_history),
            current_epoch: 0,
            max_history,
            best_test_loss: f64::INFINITY,
            epochs_since_improvement: 0,
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

        // Early stopping tracking
        if test_loss < self.best_test_loss {
            self.best_test_loss = test_loss;
            self.epochs_since_improvement = 0;
        } else {
            self.epochs_since_improvement += 1;
        }
    }

    pub fn reset(&mut self) {
        self.training_losses.clear();
        self.test_losses.clear();
        self.current_epoch = 0;
        self.best_test_loss = f64::INFINITY;
        self.epochs_since_improvement = 0;
    }

    /// Checks if early stopping should be triggered
    pub fn should_stop_early(&self, patience: usize) -> bool {
        self.epochs_since_improvement >= patience
    }

    /// Gets the training progress as a percentage from 0 to 1
    pub fn get_convergence_estimate(&self) -> f64 {
        if self.test_losses.len() < 3 {
            return 0.0;
        }

        // Calculate approximate convergence based on loss trend
        let recent_losses: Vec<_> = self.test_losses.iter().rev().take(5).collect();
        let last_loss = *recent_losses[0];

        // If we've reached a very small loss, consider training nearly complete
        if last_loss < 0.01 {
            return 0.95;
        }

        // Calculate rate of improvement
        let avg_improvement = recent_losses.windows(2)
            .map(|w| (w[1] - w[0]).abs() / w[1].max(0.0001))
            .sum::<f64>() / 4.0;

        // If improvement rate is very small, convergence is likely
        let convergence = 1.0 - (avg_improvement * 10.0).min(1.0);
        convergence.max(0.0).min(0.9)
    }
}