use crate::resources::mlp_config::MLPConfig;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub fn mlp_config_ui(mut contexts: EguiContexts, mut mlp_config: ResMut<MLPConfig>) {
    egui::Window::new("MLP Configuration").show(contexts.ctx_mut(), |ui| {
        ui.heading("MLP Hidden Layers Configuration");

        // Contrôle du nombre de couches cachées.
        let layer_count = mlp_config.hidden_layers.len();
        ui.horizontal(|ui| {
            ui.label("Number of hidden layers:");
            let mut count = layer_count as u32;
            if ui
                .add(egui::DragValue::new(&mut count).speed(1).range(0..=10))
                .changed()
            {
                if count as usize > layer_count {
                    for _ in 0..(count as usize - layer_count) {
                        mlp_config.hidden_layers.push(5);
                    }
                } else {
                    mlp_config.hidden_layers.truncate(count as usize);
                }
            }
        });

        for (i, neurons) in mlp_config.hidden_layers.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("Layer {} neuron count:", i + 1));
                ui.add(egui::DragValue::new(neurons).speed(1).range(1..=1024));
            });
        }
    });
}
