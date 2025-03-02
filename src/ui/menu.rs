use crate::states::AppState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub fn main_menu_ui(
    mut contexts: EguiContexts,
    mut next_state: ResMut<NextState<AppState>>,
) {
    let ctx = contexts.ctx_mut();

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(50.0);

            ui.heading("Machine Learning - Projet");
            ui.add_space(20.0);

            ui.label("Choisissez une option:");
            ui.add_space(30.0);

            // Boutons avec style simple
            if ui.add(egui::Button::new("Cas de Tests")
                .min_size(egui::vec2(250.0, 60.0)))
                .clicked() {
                next_state.set(AppState::CasTests);
            }

            ui.add_space(10.0);

            if ui.add(egui::Button::new("Classification de Jeux")
                .min_size(egui::vec2(250.0, 60.0)))
                .clicked() {
                next_state.set(AppState::ClassificationJeux);
            }
        });
    });
}