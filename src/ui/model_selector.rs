use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;
use crate::algorithms::mlp::{Activation, MLP};
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::DataModel;
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use bevy_egui::EguiContexts;

pub fn update_model_selector_ui(
    mut contexts: EguiContexts,
    mut training_state: ResMut<TrainingState>,
    data_model: Res<DataModel>,
) {
    egui::Window::new("Model Selector").show(contexts.ctx_mut(), |ui| {
        ui.label("Choose the model for training:");
        if ui.button("Linear Regression").clicked() {
            if data_model.is_classification() {
                ui.label("Warning: this model is for regression!");
            } else {
                training_state.selected_model = Some(ModelAlgorithm::LinearRegression(
                    LinearRegression::new(data_model.input_dim()),
                ));
            }
        }
        if ui.button("Linear Classifier").clicked() {
            if !data_model.is_classification() {
                ui.label("Warning: this model is for classification!");
            } else {
                training_state.selected_model = Some(ModelAlgorithm::LinearClassifier(
                    LinearClassifier::new(data_model.input_dim(), data_model.n_classes().unwrap()),
                ));
            }
        }
        if ui.button("MLP").clicked() {
            if data_model.is_classification() {
                training_state.selected_model = Some(ModelAlgorithm::MLP(MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    data_model.n_classes().unwrap(),
                    vec![Activation::Tanh, Activation::Tanh],
                    true,
                )));
            } else {
                training_state.selected_model = Some(ModelAlgorithm::MLP(MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    1,
                    vec![Activation::Tanh, Activation::Tanh],
                    false,
                )));
            }
        }
    });
}
