use crate::algorithms::model_selector::ModelAlgorithm;
use crate::algorithms::mlp::{Activation, MLP};
use crate::data::image_loader::ImageLoader;
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use crate::data::image_processing::{ImageDataset, ImagePreprocessor, TARGET_SIZE};

#[derive(Resource)]
pub struct ImageClassifierState {
    pub selected_image_path: Option<PathBuf>,
    pub processed_image_vec: Option<DVector<f64>>,
    pub prediction_result: Option<(usize, Vec<f64>)>,
    pub category_names: Vec<String>,
    pub texture_handle: Option<Handle<Image>>,
    pub dataset_path: Option<PathBuf>,
    pub loaded_dataset: Option<Arc<ImageDataset>>,
    pub train_progress: Option<f32>,
    pub train_message: Option<String>,
}

impl Default for ImageClassifierState {
    fn default() -> Self {
        Self {
            selected_image_path: None,
            processed_image_vec: None,
            prediction_result: None,
            category_names: vec!["FPS".to_string(), "MOBA".to_string(), "RTS".to_string(), "RPG".to_string()],
            texture_handle: None,
            dataset_path: None,
            loaded_dataset: None,
            train_progress: None,
            train_message: None,
        }
    }
}

pub fn image_classifier_ui(
    mut contexts: EguiContexts,
    mut image_state: ResMut<ImageClassifierState>,
    mut training_state: ResMut<TrainingState>,
    mut textures: ResMut<Assets<Image>>,
) {
    let mut load_image = false;
    let mut load_dataset = false;
    let mut classify_image = false;
    let mut train_model = false;

    // Variable pour stocker l'ID de la texture
    let mut texture_id = None;

    // Si une texture est définie, essayer d'obtenir son ID avant d'entrer dans la closure de UI
    if let Some(texture_handle) = &image_state.texture_handle {
        texture_id = contexts.image_id(texture_handle);
    }

    egui::Window::new("Image Classifier").show(contexts.ctx_mut(), |ui| {
        ui.heading("Classification d'images de jeux vidéo");

        ui.horizontal(|ui| {
            if ui.button("Charger une image...").clicked() {
                load_image = true;
            }

            if ui.button("Charger un dataset...").clicked() {
                load_dataset = true;
            }

            if image_state.processed_image_vec.is_some() {
                if ui.button("Classifier l'image").clicked() {
                    classify_image = true;
                }
            }
        });

        ui.separator();

        // Affichage des informations sur l'image
        if let Some(path) = &image_state.selected_image_path {
            ui.label(format!("Image: {}", path.file_name().unwrap().to_string_lossy()));

            // Afficher l'aperçu de l'image prétraitée
            if let Some(id) = texture_id {
                let size = egui::vec2(128.0, 128.0);
                ui.image((id, size));
            }
        } else {
            ui.label("Aucune image sélectionnée");
        }

        ui.separator();

        // Affichage des résultats de classification
        if let Some((class, scores)) = &image_state.prediction_result {
            ui.heading("Résultat de la classification");

            let category_name = if *class < image_state.category_names.len() {
                &image_state.category_names[*class]
            } else {
                "Catégorie inconnue"
            };

            ui.colored_label(
                egui::Color32::GREEN,
                format!("Prédiction: {} (Classe {})", category_name, class)
            );

            ui.label("Scores par catégorie:");
            for (i, &score) in scores.iter().enumerate() {
                let category = if i < image_state.category_names.len() {
                    &image_state.category_names[i]
                } else {
                    "???"
                };

                let normalized_score = (score + 1.0) / 2.0; // Normaliser de [-1,1] à [0,1]
                let progress = normalized_score.max(0.0).min(1.0) as f32;

                ui.horizontal(|ui| {
                    ui.label(format!("{}: ", category));
                    ui.add(egui::ProgressBar::new(progress).show_percentage());
                });
            }
        }

        ui.separator();

        // Affichage des informations sur le dataset
        if let Some(dataset) = &image_state.loaded_dataset {
            ui.heading("Dataset chargé");
            ui.label(format!("Nombre d'images: {}", dataset.data.len()));
            ui.label(format!("Catégories: {:?}", image_state.category_names));

            if ui.button("Entraîner un MLP").clicked() {
                train_model = true;
            }

            // Afficher la progression de l'entraînement
            if let Some(progress) = image_state.train_progress {
                ui.add(egui::ProgressBar::new(progress)
                    .show_percentage()
                    .text(image_state.train_message.clone().unwrap_or_default()));
            }
        }
    });

    // Exécuter les actions déclenchées par l'UI en dehors de la closure
    if load_image {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg"])
            .pick_file()
        {
            match ImageLoader::load_and_preprocess(&path) {
                Ok(img_vec) => {
                    let texture_handle = ImageLoader::create_preview_texture(&img_vec, &mut textures);

                    image_state.selected_image_path = Some(path);
                    image_state.processed_image_vec = Some(img_vec);
                    image_state.texture_handle = Some(texture_handle);
                    image_state.prediction_result = None;
                },
                Err(e) => {
                    println!("Erreur lors du chargement de l'image: {}", e);
                }
            }
        }
    }

    if load_dataset {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file()
        {
            match ImagePreprocessor::load_dataset(&path) {
                Ok(dataset) => {
                    // Mettre à jour les noms de catégories
                    let mut category_names = Vec::new();
                    for i in 0..dataset.reverse_mapping.len() {
                        if let Some(name) = dataset.reverse_mapping.get(&i) {
                            category_names.push(name.clone());
                        }
                    }

                    println!("Dataset chargé avec {} images et catégories: {:?}",
                             dataset.data.len(),
                             category_names);

                    image_state.category_names = category_names;
                    image_state.dataset_path = Some(path);
                    image_state.loaded_dataset = Some(Arc::new(dataset));
                },
                Err(e) => {
                    println!("Erreur lors du chargement du dataset: {}", e);
                }
            }
        }
    }

    if classify_image {
        if let (Some(img_vec), Some(model)) = (&image_state.processed_image_vec, &training_state.selected_model) {
            match model.predict(img_vec) {
                Ok(scores) => {
                    // Trouver la classe avec le score le plus élevé
                    let mut max_score = f64::NEG_INFINITY;
                    let mut predicted_class = 0;

                    let scores_vec = scores.iter().cloned().collect::<Vec<f64>>();

                    for (i, &score) in scores_vec.iter().enumerate() {
                        if score > max_score {
                            max_score = score;
                            predicted_class = i;
                        }
                    }

                    image_state.prediction_result = Some((predicted_class, scores_vec));
                },
                Err(e) => {
                    println!("Erreur lors de la prédiction: {}", e);
                }
            }
        } else {
            println!("Aucun modèle sélectionné pour la classification");
        }
    }

    if train_model {
        if let Some(dataset) = &image_state.loaded_dataset {
            // Créer un MLP adapté à notre problème de classification d'images
            let n_classes = dataset.category_mapping.len();
            let input_dim = TARGET_SIZE.0 as usize * TARGET_SIZE.1 as usize; // 64x64 = 4096

            match MLP::new(
                input_dim,
                vec![256, 64], // Architecture pyramidale pour dimensionalité élevée
                n_classes,
                vec![Activation::ReLU, Activation::ReLU, Activation::Tanh],
            ) {
                Ok(mlp) => {
                    training_state.selected_model = Some(ModelAlgorithm::new_mlp(
                        mlp,
                        true, // Classification
                    ));

                    // Diviser le dataset en train/test
                    let (train_indices, test_indices) = ImagePreprocessor::split_dataset(dataset, 0.8);

                    // Convertir en matrices
                    let (train_x, train_y) = ImagePreprocessor::dataset_to_matrices(dataset, &train_indices);

                    // Configurer l'entraînement
                    training_state.hyperparameters.learning_rate = 0.001;
                    training_state.hyperparameters.batch_size = 32;
                    training_state.hyperparameters.epoch_interval = 0.2; // Plus rapide
                    training_state.is_training = true;

                    // Message de début d'entraînement
                    image_state.train_message = Some(format!(
                        "Entraînement d'un MLP pour classifier {} images en {} classes...",
                        train_indices.len(),
                        n_classes
                    ));
                    image_state.train_progress = Some(0.0);

                    println!(
                        "Entraînement démarré sur {} images, {} dimensions, {} classes",
                        train_indices.len(),
                        input_dim,
                        n_classes
                    );
                },
                Err(e) => {
                    println!("Erreur lors de la création du MLP: {}", e);
                }
            }
        }
    }

    // Mettre à jour la progression de l'entraînement
    if training_state.is_training && image_state.train_progress.is_some() {
        let epochs = training_state.metrics.current_epoch;
        let convergence = training_state.metrics.get_convergence_estimate() as f32;

        if let Some(ref mut progress) = image_state.train_progress {
            *progress = convergence;
        }

        if let Some(ref mut message) = image_state.train_message {
            *message = format!(
                "Epoch {}: Loss = {:.6}",
                epochs,
                training_state.metrics.test_losses.back().unwrap_or(&f64::NAN)
            );
        }
    }
}