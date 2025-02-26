use crate::resources::training::TrainingState;
use crate::data::DataModel;
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::algorithms::mlp::{MLP, Activation};
use crate::algorithms::rbf::RBF;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Button, Stroke, Color32};

pub fn update_model_selector_ui(
    mut contexts: EguiContexts,
    mut training_state: ResMut<TrainingState>,
    data_model: Res<DataModel>,
) {
    egui::Window::new("Model Selector").show(contexts.ctx_mut(), |ui| {
        ui.label("Choose a model for training:");

        if !data_model.is_classification() {
            // Options for regression
            ui.horizontal(|ui| {
                ui.heading("Regression Models");
            });

            // Determine if specific model types are selected
            let is_lr = matches!(training_state.selected_model,
                Some(ModelAlgorithm::LinearRegression(_, _)));

            let is_mlp_reg = matches!(training_state.selected_model,
                Some(ModelAlgorithm::MLP(_, _)) if !training_state.selected_model.as_ref().unwrap().is_classification());

            let is_rbf_reg = matches!(training_state.selected_model,
                Some(ModelAlgorithm::RBF(_, _)) if !training_state.selected_model.as_ref().unwrap().is_classification());

            // Linear Regression button
            let lr_button = Button::new("Linear Regression")
                .stroke(if is_lr { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(lr_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::new_linear_regression(
                    data_model.input_dim(),
                ));
                println!("Selected Linear Regression model");
            }

            // MLP Regression button
            let mlp_reg_button = Button::new("MLP (Regression)")
                .stroke(if is_mlp_reg { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(mlp_reg_button).clicked() {
                match MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    1,
                    vec![Activation::Tanh, Activation::Linear], // Linear output for regression
                ) {
                    Ok(mlp) => {
                        training_state.selected_model = Some(ModelAlgorithm::new_mlp(
                            mlp,
                            false, // Not classification
                        ));
                        println!("Selected MLP model for regression");
                    },
                    Err(e) => {
                        println!("Error creating MLP model: {}", e);
                    }
                }
            }

            // RBF Regression button
            let rbf_reg_button = Button::new("RBF (Regression)")
                .stroke(if is_rbf_reg { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(rbf_reg_button).clicked() {
                let rbf = RBF::new(
                    data_model.input_dim(),
                    10,        // number of centers
                    1.0,       // gamma - reasonable default for normalized data
                    false,     // regression
                    true,      // use k-means to elect centers
                    100,       // max k-means iterations
                );
                training_state.selected_model = Some(ModelAlgorithm::new_rbf(rbf));
                println!("Selected RBF model for regression");
            }
        } else {
            // Options for classification
            ui.horizontal(|ui| {
                ui.heading("Classification Models");
            });

            let n_classes = data_model.n_classes().unwrap_or(2);

            // Determine if specific model types are selected
            let is_lc = matches!(training_state.selected_model,
                Some(ModelAlgorithm::LinearClassifier(_, _)));

            let is_mlp_class = matches!(training_state.selected_model,
                Some(ModelAlgorithm::MLP(_, _)) if training_state.selected_model.as_ref().unwrap().is_classification());

            let is_rbf_class = matches!(training_state.selected_model,
                Some(ModelAlgorithm::RBF(_, _)) if training_state.selected_model.as_ref().unwrap().is_classification());

            // Linear Classifier button
            let lc_button = Button::new("Linear Classifier")
                .stroke(if is_lc { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(lc_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::new_linear_classifier(
                    data_model.input_dim(),
                    n_classes,
                ));
                println!("Selected Linear Classifier model with {} classes", n_classes);
            }

            // MLP Classification button
            let mlp_class_button = Button::new("MLP (Classification)")
                .stroke(if is_mlp_class { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(mlp_class_button).clicked() {
                match MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    n_classes,
                    vec![Activation::Tanh, Activation::Tanh],
                ) {
                    Ok(mlp) => {
                        training_state.selected_model = Some(ModelAlgorithm::new_mlp(
                            mlp,
                            true, // Is classification
                        ));
                        println!("Selected MLP model for classification with {} classes", n_classes);
                    },
                    Err(e) => {
                        println!("Error creating MLP model: {}", e);
                    }
                }
            }

            // RBF Classification button
            let rbf_class_button = Button::new("RBF (Classification)")
                .stroke(if is_rbf_class { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(rbf_class_button).clicked() {
                let rbf = RBF::new(
                    data_model.input_dim(),
                    10,
                    1.0,
                    true,      // classification
                    true,
                    100,
                );
                training_state.selected_model = Some(ModelAlgorithm::new_rbf(rbf));
                println!("Selected RBF model for classification");
            }
        }

        // Show currently selected model
        ui.separator();
        match &training_state.selected_model {
            Some(model) => {
                let model_type = match model {
                    ModelAlgorithm::LinearRegression(_, _) => "Linear Regression",
                    ModelAlgorithm::LinearClassifier(_, _) => "Linear Classifier",
                    ModelAlgorithm::MLP(_, _) => if model.is_classification() {
                        "MLP (Classification)"
                    } else {
                        "MLP (Regression)"
                    },
                    ModelAlgorithm::RBF(_, _) => if model.is_classification() {
                        "RBF (Classification)"
                    } else {
                        "RBF (Regression)"
                    },
                };
                ui.colored_label(Color32::GREEN, format!("Selected model: {}", model_type));
            },
            None => {
                ui.colored_label(Color32::YELLOW, "No model selected");
            }
        }
    });
}