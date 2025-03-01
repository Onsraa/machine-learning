use crate::resources::mlp_config::MLPConfig;
use crate::resources::training::TrainingState;
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::algorithms::mlp::{Activation, MLP};
use crate::data::DataModel;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub fn mlp_config_ui(
    mut contexts: EguiContexts,
    mut training_state: ResMut<TrainingState>,
    mut mlp_config: ResMut<MLPConfig>,
    data_model: Res<DataModel>,
) {
    if let Some(ref model) = training_state.selected_model {
        let is_mlp = match model {
            ModelAlgorithm::MLP(_, _) => true,
            _ => false,
        };

        if is_mlp {
            let is_classification = model.is_classification();

            egui::Window::new("MLP Configuration").show(contexts.ctx_mut(), |ui| {
                ui.heading("Hidden Layer Configuration");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Model type:");
                    ui.colored_label(
                        if is_classification { egui::Color32::GOLD } else { egui::Color32::LIGHT_BLUE },
                        if is_classification { "Classification" } else { "Regression" }
                    );
                });
                ui.separator();

                ui.label("Number of hidden layers:");
                let mut count = mlp_config.hidden_layers.len() as u32;
                if ui.add(egui::DragValue::new(&mut count).speed(1).range(0..=10)).changed() {
                    println!(
                        "MLP: Changed number of hidden layers from {} to {}",
                        mlp_config.hidden_layers.len(),
                        count
                    );
                    if count > mlp_config.hidden_layers.len() as u32 {
                        for _ in 0..(count as usize - mlp_config.hidden_layers.len()) {
                            mlp_config.hidden_layers.push(5);
                        }
                    } else {
                        mlp_config.hidden_layers.truncate(count as usize);
                    }
                    println!("MLP: New hidden layer configuration: {:?}", mlp_config.hidden_layers);
                }
                ui.separator();

                for (i, neurons) in mlp_config.hidden_layers.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("Layer {} - Number of neurons:", i + 1));
                        let mut new_neurons = *neurons;
                        if ui.add(egui::DragValue::new(&mut new_neurons).speed(1).range(1..=1024)).changed() {
                            println!("MLP: Layer {}: Changed neurons from {} to {}", i + 1, *neurons, new_neurons);
                            *neurons = new_neurons;
                        }
                    });
                }

                ui.separator();
                ui.heading("Activation Functions");

                let activation_options = ["Tanh", "ReLU", "Sigmoid", "Linear"];
                let activation_values = [Activation::Tanh, Activation::ReLU, Activation::Sigmoid, Activation::Linear];

                ui.label("Hidden layers activation:");
                let mut selected_activation = activation_options.iter()
                    .position(|&a| match mlp_config.hidden_activations.first() {
                        Some(&act) => match (act, a) {
                            (Activation::Tanh, "Tanh") => true,
                            (Activation::ReLU, "ReLU") => true,
                            (Activation::Sigmoid, "Sigmoid") => true,
                            (Activation::Linear, "Linear") => true,
                            _ => false,
                        },
                        None => false,
                    })
                    .unwrap_or(0);

                egui::ComboBox::from_label("")
                    .selected_text(activation_options[selected_activation])
                    .show_ui(ui, |ui| {
                        for (i, option) in activation_options.iter().enumerate() {
                            ui.selectable_value(&mut selected_activation, i, *option);
                        }
                    });

                if selected_activation < activation_values.len() {
                    let activation = activation_values[selected_activation];
                    mlp_config.hidden_activations = vec![activation];
                }

                // Output activation is determined by the problem type
                let output_activation = if is_classification {
                    Activation::Tanh
                } else {
                    Activation::Linear
                };

                ui.label(format!("Output activation: {}", match output_activation {
                    Activation::Tanh => "Tanh",
                    Activation::ReLU => "ReLU",
                    Activation::Sigmoid => "Sigmoid",
                    Activation::Linear => "Linear",
                }));
                mlp_config.output_activation = output_activation;

                ui.separator();
                if ui.button("Apply Configuration").clicked() {
                    let mut activations = vec![mlp_config.hidden_activations[0]; mlp_config.hidden_layers.len()];
                    activations.push(mlp_config.output_activation);
                    match MLP::new(
                        data_model.input_dim(),
                        mlp_config.hidden_layers.clone(),
                        if is_classification {
                            data_model.n_classes().unwrap_or(2)
                        } else {
                            1
                        },
                        activations,
                    ) {
                        Ok(mlp) => {
                            println!("Applied new MLP configuration with {} hidden layers", mlp_config.hidden_layers.len());
                            training_state.selected_model = Some(ModelAlgorithm::new_mlp(
                                mlp,
                                is_classification
                            ));
                        },
                        Err(e) => {
                            println!("Error creating MLP with new configuration: {}", e);
                        }
                    }
                }
            });
        }
    }
}