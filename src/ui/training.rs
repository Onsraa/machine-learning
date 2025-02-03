use crate::resources::training::*;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_plot::*;

pub fn training_ui_system(mut contexts: EguiContexts, mut training_state: ResMut<TrainingState>) {
    egui::Window::new("Training Control").show(contexts.ctx_mut(), |ui| {
        ui.add_space(5.0);

        // Contrôle de l'entraînement
        if training_state.is_training {
            if ui.button("Stop Training").clicked() {
                training_state.is_training = false;
            }
        } else {
            if ui.button("Start Training").clicked() {
                training_state.is_training = true;
            }
        }
        if ui.button("Reset Training").clicked() {
            println!("Resetting training...");
            training_state.should_reset = true;
            training_state.is_training = false;
        }
        ui.add_space(5.0);
        ui.separator();
        ui.add_space(5.0);
        // Hyperparamètres
        ui.heading("Hyperparameters");
        ui.add(
            egui::Slider::new(
                &mut training_state.hyperparameters.learning_rate,
                0.0001..=0.1,
            )
            .logarithmic(true)
            .text("Learning Rate"),
        );

        ui.add(
            egui::Slider::new(&mut training_state.hyperparameters.batch_size, 1..=100)
                .text("Batch Size"),
        );

        ui.add(
            egui::Slider::new(&mut training_state.hyperparameters.train_ratio, 0.5..=0.9)
                .text("Train Ratio"),
        );

        ui.add_space(5.0);
        ui.separator();
        ui.add_space(5.0);

        // Métriques d'entraînement
        ui.heading("Training Metrics");
        ui.label(format!("Epoch: {}", training_state.metrics.current_epoch));

        if !training_state.metrics.training_losses.is_empty() {
            let last_train_loss = training_state.metrics.training_losses.back().unwrap();
            let last_test_loss = training_state.metrics.test_losses.back().unwrap();
            ui.label(format!("Training Loss: {:.6}", last_train_loss));
            ui.label(format!("Test Loss: {:.6}", last_test_loss));
        }

        // Graphique des pertes
        plot_losses(ui, &training_state.metrics);
    });
}

fn plot_losses(ui: &mut egui::Ui, metrics: &TrainingMetrics) {
    use egui_plot::{Line, Plot, PlotPoints};

    let train_points: PlotPoints = metrics
        .training_losses
        .iter()
        .enumerate()
        .map(|(i, &loss)| [i as f64, loss])
        .collect();

    let test_points: PlotPoints = metrics
        .test_losses
        .iter()
        .enumerate()
        .map(|(i, &loss)| [i as f64, loss])
        .collect();

    Plot::new("training_plot")
        .height(200.0)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(train_points).name("Training Loss"));
            plot_ui.line(Line::new(test_points).name("Test Loss"));
        });
}
