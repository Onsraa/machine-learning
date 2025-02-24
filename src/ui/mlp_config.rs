use crate::resources::mlp_config::MLPConfig;
use crate::resources::training::TrainingState;
use crate::algorithms::model_selector::ModelAlgorithm;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub fn mlp_config_ui(
    mut contexts: EguiContexts,
    training_state: Res<TrainingState>,
    mut mlp_config: ResMut<MLPConfig>,
) {
    // Afficher le panneau uniquement si le modèle sélectionné est un MLP
    let show_panel = if let Some(ModelAlgorithm::MLP(_)) = training_state.selected_model {
        true
    } else {
        false
    };

    if show_panel {
        egui::Window::new("MLP Configuration").show(contexts.ctx_mut(), |ui| {
            ui.heading("Configuration des couches cachées du MLP");

            let layer_count = mlp_config.hidden_layers.len();
            ui.horizontal(|ui| {
                ui.label("Nombre de couches cachées:");
                let mut count = layer_count as u32;
                if ui.add(egui::DragValue::new(&mut count).speed(1).range(0..=10)).changed() {
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
                    ui.label(format!("Nombre de neurones dans la couche {}:", i + 1));
                    ui.add(egui::DragValue::new(neurons).speed(1).range(1..=1024));
                });
            }
        });
    }
}
