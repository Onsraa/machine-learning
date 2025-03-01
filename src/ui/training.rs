use crate::resources::training::*;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub fn training_ui_system(mut contexts: EguiContexts, mut training_state: ResMut<TrainingState>) {
    egui::Window::new("Training Control").show(contexts.ctx_mut(), |ui| {
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            if training_state.is_training {
                if ui.button("Stop Training").clicked() {
                    println!("Stopping training...");
                    training_state.is_training = false;
                }
            } else {
                if ui.button("Start Training").clicked() {
                    println!("Starting training...");
                    training_state.is_training = true;
                }
            }
            ui.add_space(5.0);
            if ui.button("Reset Training").clicked() {
                println!("Resetting training...");
                training_state.should_reset = true;
                training_state.is_training = false;
            }
        });

        ui.add_space(5.0);
        ui.separator();
        ui.add_space(5.0);

        ui.heading("Hyperparameters");

        ui.horizontal(|ui| {
            ui.label("Learning Rate:");
            ui.add(
                egui::Slider::new(
                    &mut training_state.hyperparameters.learning_rate,
                    0.00001..=1.0,
                )
                .logarithmic(true)
                .text("learning rate"),
            );
        });
        ui.label(format!(
            "Current Value: {:.6}",
            training_state.hyperparameters.learning_rate
        ));
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Train Ratio:");
            ui.add(
                egui::Slider::new(&mut training_state.hyperparameters.train_ratio, 0.1..=1.0)
                    .text("train ratio"),
            );
        });
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Batch Size:");
            ui.add(
                egui::DragValue::new(&mut training_state.hyperparameters.batch_size)
                    .range(1..=1024),
            );
        });
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Epoch Interval (s):");
            ui.add(
                egui::Slider::new(
                    &mut training_state.hyperparameters.epoch_interval,
                    0.01..=1.0,
                )
                .logarithmic(true),
            );
        });

        ui.add_space(5.0);
        ui.separator();
        ui.add_space(5.0);

        ui.heading("Training Metrics");
        if !training_state.metrics.training_losses.is_empty() {
            ui.label(format!("Epoch: {}", training_state.metrics.current_epoch));

            if let Some(last_train) = training_state.metrics.training_losses.back() {
                ui.label(format!("Training Loss: {:.6}", last_train));
            }

            if let Some(last_test) = training_state.metrics.test_losses.back() {
                ui.label(format!("Test Loss: {:.6}", last_test));
            }

            if training_state.metrics.training_losses.len() > 1
                && training_state.metrics.test_losses.len() > 1
            {
                let test_losses: Vec<_> = training_state.metrics.test_losses.iter().collect();
                let min_test_loss = test_losses.iter().fold(f64::INFINITY, |a, &b| a.min(*b));
                let last_test_loss = *test_losses.last().unwrap_or(&&0.0);

                ui.label(format!("Best Test Loss: {:.6}", min_test_loss));

                if last_test_loss <= &(min_test_loss + 1e-6) {
                    ui.colored_label(egui::Color32::GREEN, "✓ Model is still improving");
                } else {
                    let epochs_since_best = test_losses
                        .iter()
                        .rev()
                        .position(|&loss| (loss - min_test_loss).abs() < 1e-6)
                        .unwrap_or(0);

                    ui.colored_label(
                        egui::Color32::YELLOW,
                        format!("⚠ No improvement for {} epochs", epochs_since_best),
                    );
                }
            }
        } else {
            ui.label("No training data available yet");
        }

        plot_losses_with_legend(ui, &training_state.metrics);
    });
}

fn plot_losses_with_legend(ui: &mut egui::Ui, metrics: &TrainingMetrics) {
    use egui_plot::{Legend, Line, Plot, PlotPoints};

    if metrics.training_losses.is_empty() {
        return;
    }

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
        .legend(Legend::default())
        .height(200.0)
        .y_axis_min_width(2.0)
        .allow_zoom(true)
        .allow_drag(true)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(train_points)
                    .name("Training Loss")
                    .color(egui::Color32::BLUE),
            );
            plot_ui.line(
                Line::new(test_points)
                    .name("Test Loss")
                    .color(egui::Color32::RED),
            );
        });
}
