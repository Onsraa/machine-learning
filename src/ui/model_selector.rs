// src/ui/model_selector.rs

use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;
use crate::algorithms::mlp::{Activation, MLP};
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::DataModel;
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Button, Color32, Stroke};

pub fn update_model_selector_ui(
    mut contexts: EguiContexts,
    mut training_state: ResMut<TrainingState>,
    data_model: Res<DataModel>,
) {
    egui::Window::new("Model Selector").show(contexts.ctx_mut(), |ui| {
        ui.label("Choisissez le modèle à utiliser pour l'entraînement:");

        if !data_model.is_classification() {
            // Régression
            let is_lr = matches!(
                training_state.selected_model,
                Some(ModelAlgorithm::LinearRegression(_))
            );
            let lr_button = Button::new("Linear Regression").stroke(if is_lr {
                Stroke::new(2.0, Color32::GOLD)
            } else {
                Stroke::new(1.0, Color32::LIGHT_GRAY)
            });
            if ui.add(lr_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::LinearRegression(
                    LinearRegression::new(data_model.input_dim()),
                ));
            }
            let is_mlp_reg = matches!(training_state.selected_model, Some(ModelAlgorithm::MLP(_)))
                && !data_model.is_classification();
            let mlp_reg_button = Button::new("MLP (Régression)").stroke(if is_mlp_reg {
                Stroke::new(2.0, Color32::GOLD)
            } else {
                Stroke::new(1.0, Color32::LIGHT_GRAY)
            });
            if ui.add(mlp_reg_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::MLP(MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    1,
                    vec![Activation::Tanh, Activation::Tanh],
                )));
            }
        } else {
            // Classification
            let is_lc = matches!(
                training_state.selected_model,
                Some(ModelAlgorithm::LinearClassifier(_))
            );
            let lc_button = Button::new("Linear Classifier").stroke(if is_lc {
                Stroke::new(2.0, Color32::GOLD)
            } else {
                Stroke::new(1.0, Color32::LIGHT_GRAY)
            });
            if ui.add(lc_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::LinearClassifier(
                    LinearClassifier::new(data_model.input_dim(), data_model.n_classes().unwrap()),
                ));
            }
            let is_mlp_class =
                matches!(training_state.selected_model, Some(ModelAlgorithm::MLP(_)))
                    && data_model.is_classification();
            let mlp_class_button = Button::new("MLP (Classification)").stroke(if is_mlp_class {
                Stroke::new(2.0, Color32::GOLD)
            } else {
                Stroke::new(1.0, Color32::LIGHT_GRAY)
            });
            if ui.add(mlp_class_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::MLP(MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    data_model.n_classes().unwrap(),
                    vec![Activation::Tanh, Activation::Tanh],
                )));
            }
        }
    });
}
