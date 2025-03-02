use crate::algorithms::mlp::{Activation, MLP};
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::dataset_processor::{DatasetProcessor, TARGET_SIZE};
use crate::resources::game_image_state::GameImageState;
use crate::resources::mlp_image_config::MLPImageConfig;
use crate::resources::model_managers::ModelManager;
use crate::resources::training::TrainingState;
use crate::states::TrainingState as AppTrainingState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_plot::{Line, Plot, PlotPoints};
use std::path::PathBuf;
use std::sync::Arc;

pub fn game_classifier_ui(
    mut contexts: EguiContexts,
    mut game_state: ResMut<GameImageState>,
    mut mlp_config: ResMut<MLPImageConfig>,
    mut training_state: ResMut<TrainingState>,
    mut app_training_state: ResMut<State<AppTrainingState>>,
    mut next_training_state: ResMut<NextState<AppTrainingState>>,
    mut model_manager: ResMut<ModelManager>,
) {
    if game_state.image_path_input.is_empty() {
        game_state.image_path_input = "predict_dataset/".to_string();
    }

    let ctx = contexts.ctx_mut();
    let mut load_dataset = false;
    let mut predict_image = false;
    let mut start_training = false;
    let mut stop_training = false;
    let mut save_model = false;
    let mut load_model = false;
    let mut refresh_image_models = false;

    if model_manager.confirm_delete_dialog_open {
        if let Some(index) = model_manager.model_to_delete {
            if index < model_manager.model_infos.len() {
                let model_name = model_manager.model_infos[index].name.clone();

                egui::Window::new("Confirm Deletion")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(format!(
                                "Are you sure you want to delete \"{}\"?",
                                model_name
                            ));
                            ui.label("This action cannot be undone.");

                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui.button("Cancel").clicked() {
                                    model_manager.cancel_delete();
                                }

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::RIGHT),
                                    |ui| {
                                        if ui
                                            .button(
                                                egui::RichText::new("Delete")
                                                    .color(egui::Color32::RED),
                                            )
                                            .clicked()
                                        {
                                            if let Err(e) = model_manager.confirm_delete() {
                                                game_state.train_message = format!("Error: {}", e);
                                                println!("Erreur lors de la suppression: {}", e);
                                            } else {
                                                game_state.train_message =
                                                    format!("Model {} deleted", model_name);
                                                println!("Modèle supprimé avec succès!");
                                            }
                                        }
                                    },
                                );
                            });
                        });
                    });
            } else {
                model_manager.cancel_delete();
            }
        }
    }

    egui::SidePanel::left("controls_panel").show(ctx, |ui| {
        ui.add_space(10.0);
        ui.heading("Image Classification");
        ui.separator();

        ui.collapsing("Dataset", |ui| {
            ui.horizontal(|ui| {
                ui.label("Image folder:");
                ui.text_edit_singleline(&mut game_state.dataset_folder_path);
            });

            if ui.button("Load Dataset").clicked() {
                load_dataset = true;
            }

            if game_state.dataset_loaded {
                if let Some(dataset) = &game_state.dataset {
                    ui.horizontal(|ui| {
                        ui.label("Status:");
                        ui.colored_label(egui::Color32::GREEN, "Dataset loaded");
                    });

                    ui.label(format!("Images: {}", dataset.data.len()));
                    ui.label(format!("Classes: {}", dataset.category_mapping.len()));

                    ui.label("Categories:");
                    ui.horizontal_wrapped(|ui| {
                        for (name, _) in &dataset.category_mapping {
                            ui.label(name);
                        }
                    });
                }
            } else {
                ui.colored_label(egui::Color32::YELLOW, "No dataset loaded");
            }
        });

        ui.separator();

        ui.collapsing("MLP Configuration", |ui| {
            ui.label(format!(
                "Input size: {} ({}×{})",
                mlp_config.input_size, TARGET_SIZE.0, TARGET_SIZE.1
            ));

            let mut layer_count = mlp_config.hidden_layers.len() as i32;
            ui.horizontal(|ui| {
                ui.label("Number of hidden layers:");
                if ui
                    .add(egui::DragValue::new(&mut layer_count).range(1..=10))
                    .changed()
                {
                    let current_len = mlp_config.hidden_layers.len();
                    if layer_count > current_len as i32 {
                        for _ in 0..(layer_count as usize - current_len) {
                            mlp_config.hidden_layers.push(64);
                        }
                    } else {
                        mlp_config.hidden_layers.truncate(layer_count as usize);
                    }
                }
            });

            for (i, neurons) in mlp_config.hidden_layers.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("Layer {} - Neurons:", i + 1));
                    ui.add(egui::DragValue::new(neurons).range(16..=1024));
                });
            }

            ui.horizontal(|ui| {
                ui.label("Hidden layer activation:");

                let activation_options = ["ReLU", "Tanh", "Sigmoid", "Linear"];
                let current_activation = match mlp_config.hidden_activation {
                    Activation::ReLU => 0,
                    Activation::Tanh => 1,
                    Activation::Sigmoid => 2,
                    Activation::Linear => 3,
                    _ => 0,
                };

                let mut selected_activation = current_activation;

                egui::ComboBox::from_id_salt("hidden_activation")
                    .selected_text(activation_options[current_activation])
                    .show_ui(ui, |ui| {
                        for (i, name) in activation_options.iter().enumerate() {
                            ui.selectable_value(&mut selected_activation, i, *name);
                        }
                    });

                if selected_activation != current_activation {
                    mlp_config.hidden_activation = match selected_activation {
                        0 => Activation::ReLU,
                        1 => Activation::Tanh,
                        2 => Activation::Sigmoid,
                        3 => Activation::Linear,
                        _ => Activation::ReLU,
                    };
                }
            });

            ui.label("Output activation: Tanh (fixed for classification)");

            ui.separator();

            ui.label("Hyperparameters");

            ui.horizontal(|ui| {
                ui.label("Learning rate:");
                ui.add(
                    egui::Slider::new(&mut mlp_config.learning_rate, 0.00001..=0.1)
                        .logarithmic(true),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Batch size:");
                ui.add(egui::Slider::new(&mut mlp_config.batch_size, 1..=128));
            });

            ui.horizontal(|ui| {
                ui.label("Training ratio:");
                ui.add(egui::Slider::new(&mut mlp_config.train_ratio, 0.5..=0.9));
            });

            ui.horizontal(|ui| {
                ui.label("Epoch interval:");
                ui.add(
                    egui::Slider::new(
                        &mut training_state.hyperparameters.epoch_interval,
                        0.001..=1.0,
                    )
                    .logarithmic(true),
                )
            });
        });

        ui.separator();

        ui.collapsing("Training", |ui| {
            let is_training = *app_training_state.get() == AppTrainingState::Training;
            let is_paused = !is_training
                && training_state.selected_model.is_some()
                && game_state.train_epochs > 0;

            ui.horizontal(|ui| {
                if is_training {
                    if ui.button("⏹ Stop").clicked() {
                        stop_training = true;
                    }
                } else if is_paused {
                    if ui.button("▶ Resume").clicked() {
                        if game_state.dataset_loaded {
                            start_training = true;
                        } else {
                            game_state.train_message = "Error: Dataset unavailable!".to_string();
                        }
                    }

                    // Ajouter le bouton Restart quand on est en pause
                    if ui.button("🔄 Restart").clicked() {
                        if game_state.dataset_loaded {
                            // Réinitialiser le modèle pour un nouveau départ
                            training_state.selected_model = None;
                            training_state.metrics.reset();
                            game_state.loss_history.clear();
                            game_state.train_epochs = 0;
                            game_state.train_progress = 0.0;
                            game_state.best_model_saved = false;

                            // Lancer l'entraînement
                            start_training = true;
                        } else {
                            game_state.train_message = "Error: Load a dataset first!".to_string();
                        }
                    }
                } else {
                    if ui.button("▶ Start").clicked() {
                        if game_state.dataset_loaded {
                            start_training = true;
                        } else {
                            game_state.train_message = "Error: Load a dataset first!".to_string();
                        }
                    }
                }
            });

            ui.label(&game_state.train_message);
            if game_state.train_epochs > 0 {
                ui.label(format!("Epoch: {}", game_state.train_epochs));
            }
        });

        ui.separator();

        ui.collapsing("Models", |ui| {
            ui.horizontal(|ui| {
                ui.label("Model name:");
                ui.text_edit_singleline(&mut game_state.model_name);
            });

            ui.horizontal(|ui| {
                if ui.button("💾 Save").clicked() {
                    if training_state.selected_model.is_some() {
                        save_model = true;
                    } else {
                        game_state.train_message = "Error: No model to save".to_string();
                    }
                }

                if ui.button("📂 Load").clicked() {
                    load_model = true;
                }

                if ui.button("🔄 Refresh list").clicked() {
                    refresh_image_models = true;
                }
            });

            let mut selected_index = model_manager.selected_model_index;

            ui.label("Available models:");
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .show(ui, |ui| {
                    let model_count = model_manager
                        .model_infos
                        .iter()
                        .filter(|info| {
                            info.model_type == "MLP"
                                && info.task_type == "Classification"
                                && info.category == "images_jeux"
                        })
                        .count();

                    let mut delete_index = None;

                    if model_count == 0 {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "No models found - Use 'Refresh list'",
                        );
                    } else {
                        for (i, info) in model_manager.model_infos.iter().enumerate() {
                            // Only show image classification models (category "images_jeux")
                            if info.model_type == "MLP"
                                && info.task_type == "Classification"
                                && info.category == "images_jeux"
                            {
                                let is_selected = model_manager.selected_model_index == Some(i);

                                ui.horizontal(|ui| {
                                    let text = if is_selected {
                                        egui::RichText::new(&info.name)
                                            .strong()
                                            .color(egui::Color32::LIGHT_BLUE)
                                    } else {
                                        egui::RichText::new(&info.name)
                                    };

                                    if ui.selectable_label(is_selected, text).clicked() {
                                        selected_index = Some(i);
                                    }

                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::RIGHT),
                                        |ui| {
                                            if ui.button("Delete").clicked() {
                                                delete_index = Some(i);
                                            }
                                        },
                                    );
                                });
                            }
                        }
                    }

                    if selected_index != model_manager.selected_model_index {
                        model_manager.selected_model_index = selected_index;
                    }

                    if let Some(index) = delete_index {
                        model_manager.request_delete_confirmation(index);
                    }
                });
        });

        ui.separator();

        ui.collapsing("Prediction", |ui| {
            ui.horizontal(|ui| {
                ui.label("Image path:");
                ui.text_edit_singleline(&mut game_state.image_path_input);
            });

            if ui.button("🔍 Classify image").clicked() {
                let path = PathBuf::from(&game_state.image_path_input);
                if path.exists() {
                    if training_state.selected_model.is_some() {
                        predict_image = true;
                    } else {
                        game_state.train_message = "Error: No trained model".to_string();
                    }
                } else {
                    game_state.train_message =
                        format!("Error: File not found - {}", path.display());
                }
            }
        });
    });

    egui::CentralPanel::default().show(ctx, |ui| {
        if !game_state.loss_history.is_empty() {
            ui.heading("Learning Curves");

            if let Some(&(train_loss, test_loss)) = game_state.loss_history.last() {
                ui.horizontal(|ui| {
                    ui.label(format!("Training loss: {:.6}", train_loss));
                    ui.label(format!("Test loss: {:.6}", test_loss));
                });
            }

            let train_points: PlotPoints = game_state
                .loss_history
                .iter()
                .enumerate()
                .map(|(i, &(train_loss, _))| [i as f64, train_loss])
                .collect();

            let test_points: PlotPoints = game_state
                .loss_history
                .iter()
                .enumerate()
                .map(|(i, &(_, test_loss))| [i as f64, test_loss])
                .collect();

            Plot::new("loss_plot")
                .height(300.0)
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(train_points)
                            .name("Training loss")
                            .color(egui::Color32::BLUE),
                    );

                    plot_ui.line(
                        Line::new(test_points)
                            .name("Test loss")
                            .color(egui::Color32::RED),
                    );
                });
        }

        if let Some((class_index, scores)) = &game_state.prediction_result {
            ui.add_space(20.0);
            ui.separator();
            ui.heading("Classification Result");

            let class_name = if let Some(dataset) = &game_state.dataset {
                dataset
                    .reverse_mapping
                    .get(class_index)
                    .cloned()
                    .unwrap_or_else(|| format!("Class {}", class_index))
            } else {
                format!("Class {}", class_index)
            };

            ui.colored_label(
                egui::Color32::GREEN,
                format!("Image classified as: {}", class_name),
            );

            ui.label("Scores by category:");

            for (i, &score) in scores.iter().enumerate() {
                let category_name = if let Some(dataset) = &game_state.dataset {
                    dataset
                        .reverse_mapping
                        .get(&i)
                        .cloned()
                        .unwrap_or_else(|| format!("Class {}", i))
                } else {
                    format!("Class {}", i)
                };

                // Normalize the Tanh score (-1 to 1) to (0 to 1)
                let normalized_score = (score + 1.0) / 2.0;
                let percentage = (normalized_score * 100.0).min(100.0).max(0.0);

                ui.horizontal(|ui| {
                    ui.label(format!("{}: ", category_name));
                    ui.add(
                        egui::ProgressBar::new(normalized_score as f32)
                            .text(format!("{:.1}%", percentage)),
                    );
                });
            }
        }
    });

    if load_dataset {
        match DatasetProcessor::process_dataset(
            PathBuf::from(&game_state.dataset_folder_path),
            None::<PathBuf>,
        ) {
            Ok(dataset) => {
                println!("Dataset chargé avec {} images", dataset.data.len());

                // Update the number of classes in the MLP configuration
                mlp_config.output_size = dataset.category_mapping.len();

                game_state.dataset = Some(Arc::new(dataset));
                game_state.dataset_loaded = true;
                game_state.train_message = "Dataset loaded successfully".to_string();
            }
            Err(e) => {
                println!("Erreur lors du chargement du dataset: {}", e);
                game_state.train_message = format!("Error: {}", e);
            }
        }
    }

    if predict_image {
        let path = PathBuf::from(&game_state.image_path_input);

        match DatasetProcessor::process_image(&path) {
            Ok(img_vec) => {
                game_state.processed_image = Some(img_vec);

                if let Some(model) = &training_state.selected_model {
                    match model.predict(game_state.processed_image.as_ref().unwrap()) {
                        Ok(scores) => {
                            let mut max_score = f64::NEG_INFINITY;
                            let mut predicted_class = 0;

                            let scores_vec = scores.iter().cloned().collect::<Vec<f64>>();

                            for (i, &score) in scores_vec.iter().enumerate() {
                                if score > max_score {
                                    max_score = score;
                                    predicted_class = i;
                                }
                            }

                            game_state.prediction_result = Some((predicted_class, scores_vec));
                            game_state.train_message = "Prediction succeeded".to_string();
                        }
                        Err(e) => {
                            game_state.train_message = format!("Prediction error: {}", e);
                        }
                    }
                } else {
                    game_state.train_message = "Error: No trained model available".to_string();
                }
            }
            Err(e) => {
                game_state.train_message = format!("Image processing error: {}", e);
            }
        }
    }

    if start_training {
        if let Some(dataset) = &game_state.dataset {
            // Si on n'a pas de modèle sélectionné, créer un nouveau MLP
            if training_state.selected_model.is_none() {
                let mut activations =
                    vec![mlp_config.hidden_activation; mlp_config.hidden_layers.len()];
                activations.push(Activation::Tanh); // Fixed output activation

                match MLP::new(
                    mlp_config.input_size,
                    mlp_config.hidden_layers.clone(),
                    mlp_config.output_size,
                    activations,
                ) {
                    Ok(mlp) => {
                        training_state.selected_model = Some(ModelAlgorithm::new_mlp(mlp, true));

                        // Réinitialiser les métriques seulement pour un nouveau modèle
                        training_state.metrics.reset();
                        game_state.loss_history.clear();
                        game_state.train_epochs = 0;
                        game_state.train_progress = 0.0;
                        game_state.best_model_saved = false;

                        println!("Created new MLP model");
                    }
                    Err(e) => {
                        game_state.train_message = format!("Error creating MLP: {}", e);
                        start_training = false; // Annuler le démarrage
                    }
                }
            } else {
                // Si on a déjà un modèle, on reprend l'entraînement avec celui-ci
                println!("Resuming training with existing model");
            }

            if start_training {
                // Configure training hyperparameters (toujours mettre à jour)
                training_state.hyperparameters.learning_rate = mlp_config.learning_rate;
                training_state.hyperparameters.batch_size = mlp_config.batch_size;
                training_state.hyperparameters.train_ratio = mlp_config.train_ratio;

                training_state.is_training = true;
                next_training_state.set(AppTrainingState::Training);

                game_state.train_message = "Training in progress...".to_string();
            }
        }
    }

    if stop_training {
        training_state.is_training = false;
        next_training_state.set(AppTrainingState::Idle);
        game_state.train_message = "Training stopped".to_string();
    }

    if refresh_image_models {
        println!("Rafraîchissement de la liste des modèles d'images de jeux...");
        model_manager.refresh();
        game_state.train_message = "Model list refreshed".to_string();
    }

    if save_model {
        if let Some(model) = &training_state.selected_model {
            let model_name = if game_state.model_name.trim().is_empty() {
                format!("GameClassifier_MLP_{}_classes", mlp_config.output_size)
            } else {
                game_state.model_name.clone()
            };

            let description = format!(
                "MLP for game image classification, architecture: {:?}",
                mlp_config.hidden_layers
            );

            match model_manager.save_model_with_category(
                model,
                &model_name,
                Some(description),
                "images_jeux",
            ) {
                Ok(_) => {
                    game_state.train_message = format!("Model {} saved", model_name);
                }
                Err(e) => {
                    game_state.train_message = format!("Error: {}", e);
                }
            }

            model_manager.refresh();
        }
    }

    if load_model {
        if let Some(index) = model_manager.selected_model_index {
            match model_manager.load_model(index) {
                Ok(model) => {
                    training_state.selected_model = Some(model);
                    game_state.train_message = "Model loaded successfully".to_string();

                    println!("Modèle chargé avec succès à l'index {}", index);
                }
                Err(e) => {
                    game_state.train_message = format!("Error loading: {}", e);
                    println!("Erreur lors du chargement du modèle: {}", e);
                }
            }
        } else {
            game_state.train_message = "No model selected".to_string();
        }
    }
}
