use crate::resources::training::*;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_plot::*;

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

        // Learning Rate slider
        ui.horizontal(|ui| {
            ui.label("Learning Rate:");
            ui.add(egui::Slider::new(&mut training_state.hyperparameters.learning_rate, 0.00001..=10.0)
                .text("learning rate"));
        });
        ui.add_space(5.0);

        // Train Ratio slider
        ui.horizontal(|ui| {
            ui.label("Train Ratio:");
            ui.add(egui::Slider::new(&mut training_state.hyperparameters.train_ratio, 0.1..=0.9)
                .text("train ratio"));
        });
        ui.add_space(5.0);

        // Batch Size input
        ui.horizontal(|ui| {
            ui.label("Batch Size:");
            ui.add(egui::DragValue::new(&mut training_state.hyperparameters.batch_size)
                .clamp_range(1..=1024));
        });
        ui.add_space(5.0);

        // Epoch Interval slider
        ui.horizontal(|ui| {
            ui.label("Epoch Interval (s):");
            ui.add(egui::Slider::new(&mut training_state.hyperparameters.epoch_interval, 0.01..=1.0)
                .logarithmic(true));
        });

        ui.add_space(5.0);
        ui.separator();
        ui.add_space(5.0);
        ui.heading("Training Metrics");
        if !training_state.metrics.training_losses.is_empty() {
            ui.label(format!("Epoch: {}", training_state.metrics.current_epoch));
            let last_train = training_state.metrics.training_losses.back().unwrap();
            let last_test = training_state.metrics.test_losses.back().unwrap();
            ui.label(format!("Training Loss: {:.6}", last_train));
            ui.label(format!("Test Loss: {:.6}", last_test));
        }

        plot_losses_with_legend(ui, &training_state.metrics);
    });
}

fn plot_losses_with_legend(ui: &mut egui::Ui, metrics: &TrainingMetrics) {
    use egui_plot::{Legend, Line, Plot, PlotPoints};

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
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(train_points)
                .name("Training Loss")
                .color(egui::Color32::BLUE));
            plot_ui.line(Line::new(test_points)
                .name("Test Loss")
                .color(egui::Color32::RED));
        });
}
