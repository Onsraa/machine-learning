use crate::algorithms::mlp::{Activation, MLP};
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::dataset_processor::{DatasetProcessor, TARGET_SIZE};
use crate::resources::game_image_state::GameImageState;
use crate::resources::mlp_image_config::MLPImageConfig;
use crate::resources::model_managers::ModelManager;
use crate::resources::training::TrainingState;
use crate::states::{AppState, TrainingState as AppTrainingState};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_plot::{Line, Plot, PlotPoints};
use nalgebra::DVector;
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
    let ctx = contexts.ctx_mut();

    // Variables d'action
    let mut load_dataset = false;
    let mut load_image = false;
    let mut predict_image = false;
    let mut start_training = false;
    let mut stop_training = false;
    let mut save_model = false;
    let mut load_model = false;

    // Panel gauche pour les contrôles
    egui::SidePanel::left("controls_panel").show(ctx, |ui| {
        ui.heading("Classification d'Images");
        ui.separator();

        // Section Dataset
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

        // Section Configuration MLP
        ui.collapsing("Configuration MLP", |ui| {
            ui.label(format!("Taille d'entrée: {} ({}×{})",
                             mlp_config.input_size, TARGET_SIZE.0, TARGET_SIZE.1));

            // Nombre de couches cachées
            let mut layer_count = mlp_config.hidden_layers.len() as i32;
            ui.horizontal(|ui| {
                ui.label("Nombre de couches cachées:");
                if ui.add(egui::DragValue::new(&mut layer_count).range(1..=5)).changed() {
                    let current_len = mlp_config.hidden_layers.len();
                    if layer_count > current_len as i32 {
                        // Ajouter des couches
                        for _ in 0..(layer_count as usize - current_len) {
                            mlp_config.hidden_layers.push(64);
                        }
                    } else {
                        // Retirer des couches
                        mlp_config.hidden_layers.truncate(layer_count as usize);
                    }
                }
            });

            // Configuration de chaque couche
            for (i, neurons) in mlp_config.hidden_layers.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("Couche {} - Neurones:", i + 1));
                    ui.add(egui::DragValue::new(neurons).range(16..=1024));
                });
            }

            // Activation des couches cachées
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

                egui::ComboBox::from_id_source("hidden_activation")
                    .selected_text(activation_options[current_activation])
                    .show_ui(ui, |ui| {
                        for (i, name) in activation_options.iter().enumerate() {
                            ui.selectable_value(&mut selected_activation, i, *name);
                        }
                    });

                // Si l'activation a changé, mettre à jour la configuration
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

            // Hyperparamètres
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

            ui.horizontal(|ui| {
                ui.label("Époques max:");
                ui.add(egui::Slider::new(&mut mlp_config.max_epochs, 10..=1000));
            });

            ui.horizontal(|ui| {
                ui.label("Patience arrêt précoce:");
                ui.add(egui::Slider::new(&mut mlp_config.early_stopping_patience, 5..=100));
            });
        });

        ui.separator();

        // Section Entraînement
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

            if is_training {
                ui.add(egui::ProgressBar::new(game_state.train_progress)
                    .show_percentage()
                    .animate(true));
            }

            if game_state.train_epochs > 0 {
                ui.label(format!("Époque: {}/{}", game_state.train_epochs, mlp_config.max_epochs));
            }
        });

        ui.separator();

        // Section Sauvegarde/Chargement
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
            });

            // Variable temporaire pour stocker l'index sélectionné
            let mut selected_index = model_manager.selected_model_index;

            // Liste des modèles sauvegardés
            ui.label("Modèles disponibles:");
            egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                let mut selected_index = model_manager.selected_model_index;

                for (i, info) in model_manager.model_infos.iter().enumerate() {
                    // Ne montrer que les modèles de classification d'images (catégorie "images_jeux")
                    if info.model_type == "MLP" && info.task_type == "Classification" && info.category == "images_jeux" {
                        let is_selected = model_manager.selected_model_index == Some(i);

                        let text = if is_selected {
                            egui::RichText::new(&info.name).strong().color(egui::Color32::LIGHT_BLUE)
                        } else {
                            egui::RichText::new(&info.name)
                        };

                        if ui.selectable_label(is_selected, text).clicked() {
                            selected_index = Some(i);
                        }
                    }
                }

                // Mettre à jour l'index sélectionné si changé
                if selected_index != model_manager.selected_model_index {
                    model_manager.selected_model_index = selected_index;
                }
            });

            // Mettre à jour l'index sélectionné après la boucle
            if selected_index != model_manager.selected_model_index {
                model_manager.selected_model_index = selected_index;
            }
        });

        ui.separator();

        // Section Prédiction
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

    // Panel central pour les graphiques et résultats
    egui::CentralPanel::default().show(ctx, |ui| {
        if !game_state.loss_history.is_empty() {
            ui.heading("Courbes d'apprentissage");

            // Dernières métriques
            if let Some(&(train_loss, test_loss)) = game_state.loss_history.last() {
                ui.horizontal(|ui| {
                    ui.label(format!("Erreur d'entraînement: {:.6}", train_loss));
                    ui.label(format!("Erreur de test: {:.6}", test_loss));
                });
            }

            // Courbe d'apprentissage
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

        // Affichage des résultats de prédiction
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

    // Exécuter les actions déclenchées par l'UI
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
        // Charger et traiter l'image
        let path = PathBuf::from(&game_state.image_path_input);

        match DatasetProcessor::process_image(&path) {
            Ok(img_vec) => {
                game_state.processed_image = Some(img_vec);

                // Prédire si un modèle est disponible
                if let Some(model) = &training_state.selected_model {
                    match model.predict(game_state.processed_image.as_ref().unwrap()) {
                        Ok(scores) => {
                            // Trouver la classe prédite
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
            // Créer un MLP adapté à la classification d'images
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
                        true, // Classification
                    ));

                    // Configurer l'entraînement
                    training_state.hyperparameters.learning_rate = mlp_config.learning_rate;
                    training_state.hyperparameters.batch_size = mlp_config.batch_size;
                    training_state.hyperparameters.train_ratio = mlp_config.train_ratio;
                    training_state.hyperparameters.early_stopping_patience = mlp_config.early_stopping_patience;

                    // Réinitialiser les métriques
                    training_state.metrics.reset();
                    game_state.loss_history.clear();
                    game_state.train_epochs = 0;
                    game_state.train_progress = 0.0;
                    game_state.best_model_saved = false;

                    // Démarrer l'entraînement
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

    // Dans la partie exécutant les actions
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

            // Utiliser la nouvelle méthode avec la catégorie "images_jeux"
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

            // Recharger la liste des modèles
            model_manager.refresh();
        }
    }

    if load_model {
        if let Some(index) = model_manager.selected_model_index {
            match model_manager.load_model(index) {
                Ok(model) => {
                    training_state.selected_model = Some(model);
                    game_state.train_message = "Modèle chargé avec succès".to_string();
                },
                Err(e) => {
                    game_state.train_message = format!("Erreur lors du chargement: {}", e);
                }
            }
        } else {
            game_state.train_message = "Aucun modèle sélectionné".to_string();
        }
    }
}