use crate::algorithms::mlp::{Activation, MLP};
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::algorithms::rbf::RBF;
use crate::algorithms::svm::{KernelType, SVM};
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
        ui.label("Choose a model for training:");

        if !data_model.is_classification() {
            ui.horizontal(|ui| {
                ui.heading("Regression Models");
            });

            let is_lr = matches!(training_state.selected_model,
                Some(ModelAlgorithm::LinearRegression(_, _)));

            let is_mlp_reg = matches!(training_state.selected_model,
                Some(ModelAlgorithm::MLP(_, _)) if !training_state.selected_model.as_ref().unwrap().is_classification());

            let is_rbf_reg = matches!(training_state.selected_model,
                Some(ModelAlgorithm::RBF(_, _)) if !training_state.selected_model.as_ref().unwrap().is_classification());

            let lr_button = Button::new("Linear Regression")
                .stroke(if is_lr { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(lr_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::new_linear_regression(
                    data_model.input_dim(),
                ));
                println!("Selected Linear Regression model");
            }

            let mlp_reg_button = Button::new("MLP (Regression)")
                .stroke(if is_mlp_reg { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(mlp_reg_button).clicked() {
                match MLP::new(
                    data_model.input_dim(),
                    vec![5],
                    1,
                    vec![Activation::Tanh, Activation::Linear],
                ) {
                    Ok(mlp) => {
                        training_state.selected_model = Some(ModelAlgorithm::new_mlp(
                            mlp,
                            false,
                        ));
                        println!("Selected MLP model for regression");
                    },
                    Err(e) => {
                        println!("Error creating MLP model: {}", e);
                    }
                }
            }

            let rbf_reg_button = Button::new("RBF (Regression)")
                .stroke(if is_rbf_reg { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(rbf_reg_button).clicked() {
                let rbf = RBF::new(
                    data_model.input_dim(),
                    10,
                    1.0,
                    false,
                    true,
                    100,
                );
                training_state.selected_model = Some(ModelAlgorithm::new_rbf(rbf));
                println!("Selected RBF model for regression");
            }
        } else {
            ui.horizontal(|ui| {
                ui.heading("Classification Models");
            });

            let n_classes = data_model.n_classes().unwrap_or(2);

            let is_lc = matches!(training_state.selected_model,
                Some(ModelAlgorithm::LinearClassifier(_, _)));

            let is_mlp_class = matches!(training_state.selected_model,
                Some(ModelAlgorithm::MLP(_, _)) if training_state.selected_model.as_ref().unwrap().is_classification());

            let is_rbf_class = matches!(training_state.selected_model,
                Some(ModelAlgorithm::RBF(_, _)) if training_state.selected_model.as_ref().unwrap().is_classification());

            let is_svm = matches!(training_state.selected_model,
                Some(ModelAlgorithm::SVM(_, _)));

            let lc_button = Button::new("Linear Classifier")
                .stroke(if is_lc { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(lc_button).clicked() {
                training_state.selected_model = Some(ModelAlgorithm::new_linear_classifier(
                    data_model.input_dim(),
                    n_classes,
                ));
                println!("Selected Linear Classifier model with {} classes", n_classes);
            }

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
                            true,
                        ));
                        println!("Selected MLP model for classification with {} classes", n_classes);
                    },
                    Err(e) => {
                        println!("Error creating MLP model: {}", e);
                    }
                }
            }

            let rbf_class_button = Button::new("RBF (Classification)")
                .stroke(if is_rbf_class { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
            if ui.add(rbf_class_button).clicked() {
                let rbf = RBF::new(
                    data_model.input_dim(),
                    10,
                    1.0,
                    true,
                    true,
                    100,
                );
                training_state.selected_model = Some(ModelAlgorithm::new_rbf(rbf));
                println!("Selected RBF model for classification");
            }

            if n_classes == 2 {
                let svm_button = Button::new("SVM (Classification)")
                    .stroke(if is_svm { Stroke::new(2.0, Color32::GREEN) } else { Stroke::new(0.5, Color32::LIGHT_GRAY) });
                if ui.add(svm_button).clicked() {
                    let svm = SVM::new(
                        data_model.input_dim(),
                        KernelType::RBF,
                        2,
                        1.0,
                        1.0,
                        1e-3,
                        1000,
                    );
                    training_state.selected_model = Some(ModelAlgorithm::new_svm(svm));
                    println!("Selected SVM model for binary classification");
                }
            } else {
                ui.label("SVM only supports binary classification");
            }
        }

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
                    ModelAlgorithm::SVM(_, _) => "SVM (Classification)",
                };
                ui.colored_label(Color32::GREEN, format!("Selected model: {}", model_type));
            },
            None => {
                ui.colored_label(Color32::YELLOW, "No model selected");
            }
        }
    });
}
