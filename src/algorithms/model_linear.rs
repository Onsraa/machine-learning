use crate::data::models::{DataModel, Point, Points};
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui_plot::{Line, Plot, PlotPoints};
use rand::seq::SliceRandom;
use rand::Rng;
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrainingState {
    #[default]
    Stopped,
    Running,
}

#[derive(Resource)]
pub struct TrainingParameters {
    pub learning_rate: f32,
}

impl Default for TrainingParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
        }
    }
}

#[derive(Resource, Default)]
pub struct TrainingMetrics {
    pub training_loss: Vec<f32>,
    pub test_loss: Vec<f32>,
    pub current_epoch: usize,
}

#[derive(Resource)]
pub struct LinearModel {
    weights: Vec<f32>,
    bias: f32,
}

impl Default for LinearModel {
    fn default() -> Self {
        Self::new(2)
    }
}

impl LinearModel {
    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: rng.gen_range(-1.0..1.0),
        }
    }

    pub fn predict(&self, input: &[f32]) -> f32 {
        input
            .iter()
            .zip(self.weights.iter())
            .map(|(&x, &w)| x * w)
            .sum::<f32>()
            + self.bias
    }

    pub fn train_step(&mut self, batch: &[(Vec<f32>, f32)], learning_rate: f32) -> f32 {
        let batch_size = batch.len() as f32;
        let mut grad_weights = vec![0.0; self.weights.len()];
        let mut grad_bias = 0.0;
        let mut total_loss = 0.0;

        for (input, target) in batch {
            let prediction = self.predict(input);
            let error = prediction - target;
            total_loss += error * error;

            for (i, &x) in input.iter().enumerate() {
                grad_weights[i] += error * x;
            }
            grad_bias += error;
        }

        // Mise à jour des poids et du biais
        for i in 0..self.weights.len() {
            self.weights[i] -= learning_rate * grad_weights[i] / batch_size;
        }
        self.bias -= learning_rate * grad_bias / batch_size;

        total_loss / batch_size
    }
}

pub fn training_ui(
    mut contexts: EguiContexts,
    mut params: ResMut<TrainingParameters>,
    training_state: Res<State<TrainingState>>,
    mut next_state: ResMut<NextState<TrainingState>>,
    metrics: Res<TrainingMetrics>,
) {
    egui::Window::new("Training Control").show(contexts.ctx_mut(), |ui| {
        ui.group(|ui| {
            ui.label("Hyperparameter");
            ui.add(
                egui::Slider::new(&mut params.learning_rate, 0.0001..=0.1)
                    .logarithmic(true)
                    .text("Learning Rate"),
            );
        });

        ui.horizontal(|ui| match training_state.get() {
            TrainingState::Stopped => {
                if ui.button("Start Training").clicked() {
                    next_state.set(TrainingState::Running);
                }
            }
            TrainingState::Running => {
                if ui.button("Stop Training").clicked() {
                    next_state.set(TrainingState::Stopped);
                }
            }
        });

        Plot::new("losses").height(200.0).show(ui, |plot_ui| {
            if !metrics.training_loss.is_empty() {
                let training_line = Line::new(PlotPoints::from_iter(
                    metrics
                        .training_loss
                        .iter()
                        .enumerate()
                        .map(|(i, &loss)| [i as f64, loss as f64]),
                ))
                    .name("Training Loss");

                let test_line = Line::new(PlotPoints::from_iter(
                    metrics
                        .test_loss
                        .iter()
                        .enumerate()
                        .map(|(i, &loss)| [i as f64, loss as f64]),
                ))
                    .name("Test Loss");

                plot_ui.line(training_line);
                plot_ui.line(test_line);
            }
        });
    });
}

#[derive(Resource)]
pub struct TrainingData {
    data: Vec<(Vec<f32>, f32)>, // (input, target)
}

impl TrainingData {
    pub fn new(data: Vec<(Vec<f32>, f32)>) -> Self {
        Self { data }
    }

    pub fn split_train_test(&self, ratio: f32) -> (Vec<(Vec<f32>, f32)>, Vec<(Vec<f32>, f32)>) {
        let split_idx = (self.data.len() as f32 * ratio) as usize;
        let mut data = self.data.clone();
        data.shuffle(&mut rand::thread_rng());

        let test = data.split_off(split_idx);
        (data, test)
    }
}

pub fn convert_points_to_training_data(mut commands: Commands, points: Res<Points>) {
    let data: Vec<(Vec<f32>, f32)> = points
        .data
        .iter()
        .map(|Point(x, y, z, _)| (vec![*x, *y], *z))
        .collect();

    commands.insert_resource(TrainingData::new(data));
}

pub fn train_epoch_system(
    mut model: ResMut<LinearModel>,
    mut metrics: ResMut<TrainingMetrics>,
    params: Res<TrainingParameters>,
    training_data: Res<TrainingData>,
    training_state: Res<State<TrainingState>>,
    mut next_training_state: ResMut<NextState<TrainingState>>,
) {
    match training_state.get() {
        TrainingState::Running => return,
        _ => {}
    }

    // Valeurs fixes au lieu de paramètres variables
    const MAX_EPOCHS: usize = 100;
    const BATCH_SIZE: usize = 32;
    const TRAINING_RATIO: f32 = 0.8;

    if metrics.current_epoch >= MAX_EPOCHS {
        next_training_state.set(TrainingState::Stopped);
        return;
    }

    // Split data avec ratio fixe
    let (train_data, test_data) = training_data.split_train_test(TRAINING_RATIO);

    // Training avec batch size fixe
    let mut epoch_loss = 0.0;
    for batch in train_data.chunks(BATCH_SIZE) {
        epoch_loss += model.train_step(batch, params.learning_rate);
    }
    epoch_loss /= (train_data.len() / BATCH_SIZE) as f32;

    // Testing
    let test_loss: f32 = test_data
        .iter()
        .map(|(input, target)| {
            let pred = model.predict(input);
            (pred - target).powi(2)
        })
        .sum::<f32>()
        / test_data.len() as f32;

    metrics.training_loss.push(epoch_loss);
    metrics.test_loss.push(test_loss);
    metrics.current_epoch += 1;
}

pub fn setup_model(mut commands: Commands, data_model: Res<DataModel>, points: Res<Points>) {
    let data: Vec<(Vec<f32>, f32)> = match *data_model {
        // Cas 2D - Prédire y à partir de x
        DataModel::LinearSimple | DataModel::LinearSimple2d => points
            .data
            .iter()
            .map(|Point(x, y, _, _)| (vec![*x], *y))
            .collect(),

        // Cas 3D - Prédire z à partir de x et y
        DataModel::LinearSimple3d | DataModel::LinearTricky3d => points
            .data
            .iter()
            .map(|Point(x, y, z, _)| (vec![*x, *y], *z))
            .collect(),

        // Pour tous les autres cas, on utilise le même format que LinearSimple
        _ => points
            .data
            .iter()
            .map(|Point(x, y, _, _)| (vec![*x], *y))
            .collect(),
    };

    // Déterminer les dimensions selon le modèle
    let input_dim = match *data_model {
        DataModel::LinearSimple3d | DataModel::LinearTricky3d => 2, // x, y comme entrées
        _ => 1,                                                     // seulement x comme entrée
    };

    // Initialiser toutes les ressources nécessaires
    commands.insert_resource(TrainingData::new(data));
    commands.insert_resource(LinearModel::new(input_dim));
    commands.insert_resource(TrainingMetrics::default());
    commands.insert_resource(TrainingParameters::default());
}
