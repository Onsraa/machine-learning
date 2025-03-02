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

                egui::Window::new("Confirmer la suppression")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.heading("Confirmer la suppression");
                            ui.label(format!(
                                "Êtes-vous sûr de vouloir supprimer \"{}\" ?",
                                model_name
                            ));
                            ui.label("Cette action ne peut pas être annulée.");

                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui.button("Annuler").clicked() {
                                    model_manager.cancel_delete();
                                }

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::RIGHT),
                                    |ui| {
                                        if ui
                                            .button(egui::RichText::new("Supprimer").color(egui::Color32::RED))
                                            .clicked()
                                        {
                                            if let Err(e) = model_manager.confirm_delete() {
                                                game_state.train_message = format!("Erreur: {}", e);
                                                println!("Erreur lors de la suppression: {}", e);
                                            } else {
                                                game_state.train_message = format!("Modèle {} supprimé", model_name);
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
        ui.heading("Classification d'Images");
        ui.separator();

        ui.collapsing("Dataset", |ui| {
            ui.horizontal(|ui| {
                ui.label("Dossier des images:");
                ui.text_edit_singleline(&mut game_state.dataset_folder_path);
            });

            if ui.button("Charger Dataset").clicked() {
                load_dataset = true;
            }

            if game_state.dataset_loaded {
                if let Some(dataset) = &game_state.dataset {
                    ui.horizontal(|ui| {
                        ui.label("Statut:");
                        ui.colored_label(egui::Color32::GREEN, "Dataset chargé");
                    });

                    ui.label(format!("Images: {}", dataset.data.len()));
                    ui.label(format!("Classes: {}", dataset.category_mapping.len()));

                    ui.label("Catégories:");
                    ui.horizontal_wrapped(|ui| {
                        for (name, _) in &dataset.category_mapping {
                            ui.label(name);
                        }
                    });
                }
            } else {
                ui.colored_label(egui::Color32::YELLOW, "Aucun dataset chargé");
            }
        });

        ui.separator();

        ui.collapsing("Configuration MLP", |ui| {
            ui.label(format!("Taille d'entrée: {} ({}×{})",
                             mlp_config.input_size, TARGET_SIZE.0, TARGET_SIZE.1));

            let mut layer_count = mlp_config.hidden_layers.len() as i32;
            ui.horizontal(|ui| {
                ui.label("Nombre de couches cachées:");
                if ui.add(egui::DragValue::new(&mut layer_count).range(1..=10)).changed() {
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
                    ui.label(format!("Couche {} - Neurones:", i + 1));
                    ui.add(egui::DragValue::new(neurons).range(16..=1024));
                });
            }

            ui.horizontal(|ui| {
                ui.label("Activation couches cachées:");

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

            ui.label("Activation sortie: Tanh (fixe pour classification)");

            ui.separator();

            ui.label("Hyperparamètres");

            ui.horizontal(|ui| {
                ui.label("Taux d'apprentissage:");
                ui.add(egui::Slider::new(&mut mlp_config.learning_rate, 0.00001..=0.1)
                    .logarithmic(true));
            });

            ui.horizontal(|ui| {
                ui.label("Taille du batch:");
                ui.add(egui::Slider::new(&mut mlp_config.batch_size, 1..=128));
            });

            ui.horizontal(|ui| {
                ui.label("Ratio d'entraînement:");
                ui.add(egui::Slider::new(&mut mlp_config.train_ratio, 0.5..=0.9));
            });
        });

        ui.separator();

        ui.collapsing("Entraînement", |ui| {
            let is_training = *app_training_state.get() == AppTrainingState::Training;

            ui.horizontal(|ui| {
                if is_training {
                    if ui.button("⏹ Arrêter").clicked() {
                        stop_training = true;
                    }
                } else {
                    if ui.button("▶ Démarrer").clicked() {
                        if game_state.dataset_loaded {
                            start_training = true;
                        } else {
                            game_state.train_message = "Erreur: Chargez d'abord un dataset!".to_string();
                        }
                    }
                }
            });
            ui.label(&game_state.train_message);
            if game_state.train_epochs > 0 {
                ui.label(format!("Époque: {}", game_state.train_epochs));
            }
        });

        ui.separator();

        ui.collapsing("Modèles", |ui| {
            ui.horizontal(|ui| {
                ui.label("Nom du modèle:");
                ui.text_edit_singleline(&mut game_state.model_name);
            });

            ui.horizontal(|ui| {
                if ui.button("💾 Sauvegarder").clicked() {
                    if training_state.selected_model.is_some() {
                        save_model = true;
                    } else {
                        game_state.train_message = "Erreur: Aucun modèle à sauvegarder".to_string();
                    }
                }

                if ui.button("📂 Charger").clicked() {
                    load_model = true;
                }

                if ui.button("🔄 Rafraîchir liste").clicked() {
                    refresh_image_models = true;
                }
            });

            let mut selected_index = model_manager.selected_model_index;

            ui.label("Modèles disponibles:");
            egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                let model_count = model_manager.model_infos.iter()
                    .filter(|info| info.model_type == "MLP" &&
                        info.task_type == "Classification" &&
                        info.category == "images_jeux")
                    .count();

                let mut delete_index = None;

                if model_count == 0 {
                    ui.colored_label(egui::Color32::YELLOW, "Aucun modèle trouvé - Utilisez 'Rafraîchir liste'");
                } else {
                    for (i, info) in model_manager.model_infos.iter().enumerate() {
                        // Ne montrer que les modèles de classification d'images (catégorie "images_jeux")
                        if info.model_type == "MLP" && info.task_type == "Classification" && info.category == "images_jeux" {
                            let is_selected = model_manager.selected_model_index == Some(i);

                            ui.horizontal(|ui| {
                                let text = if is_selected {
                                    egui::RichText::new(&info.name).strong().color(egui::Color32::LIGHT_BLUE)
                                } else {
                                    egui::RichText::new(&info.name)
                                };

                                if ui.selectable_label(is_selected, text).clicked() {
                                    selected_index = Some(i);
                                }

                                ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                                    if ui.button("🗑️ Supprimer").clicked() {
                                        delete_index = Some(i);
                                    }
                                });
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

        ui.collapsing("Prédiction", |ui| {
            ui.horizontal(|ui| {
                ui.label("Chemin de l'image:");
                ui.text_edit_singleline(&mut game_state.image_path_input);
            });

            if ui.button("🔍 Classifier l'image").clicked() {
                let path = PathBuf::from(&game_state.image_path_input);
                if path.exists() {
                    if training_state.selected_model.is_some() {
                        predict_image = true;
                    } else {
                        game_state.train_message = "Erreur: Aucun modèle entraîné".to_string();
                    }
                } else {
                    game_state.train_message = format!("Erreur: Fichier non trouvé - {}", path.display());
                }
            }
        });
    });

    egui::CentralPanel::default().show(ctx, |ui| {
        if !game_state.loss_history.is_empty() {
            ui.heading("Courbes d'apprentissage");

            if let Some(&(train_loss, test_loss)) = game_state.loss_history.last() {
                ui.horizontal(|ui| {
                    ui.label(format!("Erreur d'entraînement: {:.6}", train_loss));
                    ui.label(format!("Erreur de test: {:.6}", test_loss));
                });
            }

            let train_points: PlotPoints = game_state.loss_history
                .iter()
                .enumerate()
                .map(|(i, &(train_loss, _))| [i as f64, train_loss])
                .collect();

            let test_points: PlotPoints = game_state.loss_history
                .iter()
                .enumerate()
                .map(|(i, &(_, test_loss))| [i as f64, test_loss])
                .collect();

            Plot::new("loss_plot")
                .height(300.0)
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(train_points)
                        .name("Erreur d'entraînement")
                        .color(egui::Color32::BLUE));

                    plot_ui.line(Line::new(test_points)
                        .name("Erreur de test")
                        .color(egui::Color32::RED));
                });
        }

        if let Some((class_index, scores)) = &game_state.prediction_result {
            ui.add_space(20.0);
            ui.separator();
            ui.heading("Résultat de la Classification");

            let class_name = if let Some(dataset) = &game_state.dataset {
                dataset.reverse_mapping.get(class_index)
                    .cloned()
                    .unwrap_or_else(|| format!("Classe {}", class_index))
            } else {
                format!("Classe {}", class_index)
            };

            ui.colored_label(
                egui::Color32::GREEN,
                format!("Image classifiée comme: {}", class_name)
            );

            ui.label("Scores par catégorie:");

            for (i, &score) in scores.iter().enumerate() {
                let category_name = if let Some(dataset) = &game_state.dataset {
                    dataset.reverse_mapping.get(&i)
                        .cloned()
                        .unwrap_or_else(|| format!("Classe {}", i))
                } else {
                    format!("Classe {}", i)
                };

                // Normaliser le score de Tanh (-1 à 1) à (0 à 1)
                let normalized_score = (score + 1.0) / 2.0;
                let percentage = (normalized_score * 100.0).min(100.0).max(0.0);

                ui.horizontal(|ui| {
                    ui.label(format!("{}: ", category_name));
                    ui.add(egui::ProgressBar::new(normalized_score as f32)
                        .text(format!("{:.1}%", percentage)));
                });
            }
        }
    });

    if load_dataset {
        match DatasetProcessor::process_dataset(PathBuf::from(&game_state.dataset_folder_path), None::<PathBuf>) {
            Ok(dataset) => {
                println!("Dataset chargé avec {} images", dataset.data.len());

                // Mettre à jour le nombre de classes dans la configuration du MLP
                mlp_config.output_size = dataset.category_mapping.len();

                game_state.dataset = Some(Arc::new(dataset));
                game_state.dataset_loaded = true;
                game_state.train_message = "Dataset chargé avec succès".to_string();
            },
            Err(e) => {
                println!("Erreur lors du chargement du dataset: {}", e);
                game_state.train_message = format!("Erreur: {}", e);
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
                            game_state.train_message = "Prédiction réussie".to_string();
                        },
                        Err(e) => {
                            game_state.train_message = format!("Erreur de prédiction: {}", e);
                        }
                    }
                } else {
                    game_state.train_message = "Erreur: Aucun modèle entraîné disponible".to_string();
                }
            },
            Err(e) => {
                game_state.train_message = format!("Erreur de traitement de l'image: {}", e);
            }
        }
    }

    if start_training {
        if let Some(dataset) = &game_state.dataset {
            let mut activations = vec![mlp_config.hidden_activation; mlp_config.hidden_layers.len()];
            activations.push(Activation::Tanh); // Activation de sortie fixe

            match MLP::new(
                mlp_config.input_size,
                mlp_config.hidden_layers.clone(),
                mlp_config.output_size,
                activations,
            ) {
                Ok(mlp) => {
                    training_state.selected_model = Some(ModelAlgorithm::new_mlp(
                        mlp,
                        true,
                    ));

                    // Configurer l'entraînement
                    training_state.hyperparameters.learning_rate = mlp_config.learning_rate;
                    training_state.hyperparameters.batch_size = mlp_config.batch_size;
                    training_state.hyperparameters.train_ratio = mlp_config.train_ratio;
                    training_state.metrics.reset();
                    game_state.loss_history.clear();
                    game_state.train_epochs = 0;
                    game_state.train_progress = 0.0;
                    game_state.best_model_saved = false;

                    training_state.is_training = true;
                    next_training_state.set(AppTrainingState::Training);

                    game_state.train_message = "Entraînement en cours...".to_string();
                },
                Err(e) => {
                    game_state.train_message = format!("Erreur lors de la création du MLP: {}", e);
                }
            }
        }
    }

    if stop_training {
        training_state.is_training = false;
        next_training_state.set(AppTrainingState::Idle);
        game_state.train_message = "Entraînement arrêté".to_string();
    }

    if refresh_image_models {
        println!("Rafraîchissement de la liste des modèles d'images de jeux...");
        model_manager.refresh();
        game_state.train_message = "Liste des modèles rafraîchie".to_string();
    }

    if save_model {
        if let Some(model) = &training_state.selected_model {
            let model_name = if game_state.model_name.trim().is_empty() {
                format!("GameClassifier_MLP_{}_classes", mlp_config.output_size)
            } else {
                game_state.model_name.clone()
            };

            let description = format!(
                "MLP pour classification d'images de jeux, architecture: {:?}",
                mlp_config.hidden_layers
            );

            match model_manager.save_model_with_category(
                model,
                &model_name,
                Some(description),
                "images_jeux"
            ) {
                Ok(_) => {
                    game_state.train_message = format!("Modèle {} sauvegardé", model_name);
                },
                Err(e) => {
                    game_state.train_message = format!("Erreur: {}", e);
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
                    game_state.train_message = "Modèle chargé avec succès".to_string();

                    println!("Modèle chargé avec succès à l'index {}", index);
                },
                Err(e) => {
                    game_state.train_message = format!("Erreur lors du chargement: {}", e);
                    println!("Erreur lors du chargement du modèle: {}", e);
                }
            }
        } else {
            game_state.train_message = "Aucun modèle sélectionné".to_string();
        }
    }
}