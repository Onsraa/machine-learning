use crate::data::dataset_processor::DatasetProcessor;
use crate::resources::game_image_state::GameImageState;
use crate::resources::model_managers::ModelManager;
use crate::resources::training::TrainingState;
use crate::states::TrainingState as AppTrainingState;
use bevy::prelude::*;
use rand::prelude::*;

pub fn image_training_system(
    mut game_state: ResMut<GameImageState>,
    mut training_state: ResMut<TrainingState>,
    app_training_state: Res<State<AppTrainingState>>,
    mut next_training_state: ResMut<NextState<AppTrainingState>>,
    mut model_manager: ResMut<ModelManager>,
    time: Res<Time>,
) {
    // Ne pas continuer si on n'est pas en mode entraînement
    if *app_training_state.get() != AppTrainingState::Training || !training_state.is_training {
        return;
    }

    // Vérifier que nous avons un dataset et un modèle
    if game_state.dataset.is_none() || training_state.selected_model.is_none() {
        next_training_state.set(AppTrainingState::Idle);
        training_state.is_training = false;
        game_state.train_message = "Erreur: Dataset ou modèle non défini".to_string();
        return;
    }

    // Gestion de l'intervalle entre les époques
    let current_time = time.elapsed_secs();
    if current_time - training_state.last_update < training_state.hyperparameters.epoch_interval {
        return;
    }
    training_state.last_update = current_time;

    // Incrémenter le compteur d'époques
    game_state.train_epochs += 1;

    // Extraire les données nécessaires pour tout le traitement
    let train_ratio = training_state.hyperparameters.train_ratio;
    let batch_size = training_state.hyperparameters.batch_size;
    let learning_rate = training_state.hyperparameters.learning_rate;

    // Extraire des métadonnées du dataset que nous utiliserons plus tard
    let num_classes;
    let dataset_size;

    // Préparation des données pour l'entraînement
    let batch_inputs;
    let batch_targets;
    let test_inputs;
    let test_targets;

    {
        // Scope limité pour l'emprunt immutable du dataset
        let dataset = game_state.dataset.as_ref().unwrap();
        num_classes = dataset.category_mapping.len();
        dataset_size = dataset.data.len();

        // Diviser le dataset en ensembles d'entraînement et de test
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Réutiliser les indices si disponibles, sinon créer une nouvelle division
        if training_state.metrics.training_losses.is_empty() {
            let (new_train_idx, new_test_idx) = DatasetProcessor::split_dataset(dataset, train_ratio);
            train_indices = new_train_idx;
            test_indices = new_test_idx;

            // Mettre en cache pour les époques suivantes
            training_state.index_cache = Some((train_indices.clone(), test_indices.clone()));
        } else {
            // Utiliser les indices de l'époque précédente
            if let Some(ref idx_cache) = training_state.index_cache {
                train_indices = idx_cache.0.clone();
                test_indices = idx_cache.1.clone();
            } else {
                let (new_train_idx, new_test_idx) = DatasetProcessor::split_dataset(dataset, train_ratio);
                train_indices = new_train_idx;
                test_indices = new_test_idx;

                // Mettre en cache pour les époques suivantes
                training_state.index_cache = Some((train_indices.clone(), test_indices.clone()));
            }
        }

        // Préparer un batch pour l'entraînement
        let mut batch_indices = Vec::with_capacity(batch_size);
        let mut rng = thread_rng();

        if train_indices.len() <= batch_size {
            batch_indices = train_indices;
        } else {
            batch_indices = train_indices
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect();
        }

        // Convertir les données en matrices
        let matrices = DatasetProcessor::dataset_to_matrices(dataset, &batch_indices);
        batch_inputs = matrices.0;
        batch_targets = matrices.1;

        let test_matrices = DatasetProcessor::dataset_to_matrices(dataset, &test_indices);
        test_inputs = test_matrices.0;
        test_targets = test_matrices.1;
    }

    // Effectuer une étape d'entraînement
    if let Some(mut model) = training_state.selected_model.take() {
        match model.fit(&batch_inputs, &batch_targets, learning_rate, 1) {
            Ok(train_loss) => {
                // Évaluer sur les données de test
                match model.evaluate(&test_inputs, &test_targets) {
                    Ok(test_loss) => {
                        // Mettre à jour les métriques
                        training_state.metrics.add_metrics(train_loss, test_loss);

                        // Enregistrer l'historique pour l'affichage
                        game_state.loss_history.push((train_loss, test_loss));

                        // Mettre à jour le message d'état
                        game_state.train_message = format!(
                            "Époque {}: Train Loss = {:.6}, Test Loss = {:.6}",
                            game_state.train_epochs,
                            train_loss,
                            test_loss
                        );

                        // Mettre à jour la barre de progression
                        let convergence = training_state.metrics.get_convergence_estimate() as f32;
                        game_state.train_progress = convergence;

                        // Sauvegarder le meilleur modèle
                        if test_loss <= training_state.metrics.best_test_loss && !game_state.best_model_saved {
                            // Sauvegarder le modèle
                            let model_name = format!("GameClassifier_MLP_{}_classes", num_classes);
                            let description = format!(
                                "MLP pour classification d'images de jeux entraîné sur {} images.",
                                dataset_size
                            );

                            if let Err(e) = model_manager.save_model(
                                &model,
                                &model_name,
                                Some(description)
                            ) {
                                println!("Erreur lors de la sauvegarde du modèle: {}", e);
                            } else {
                                game_state.best_model_saved = true;
                                game_state.train_message = format!(
                                    "{} - Meilleur modèle sauvegardé!",
                                    game_state.train_message
                                );
                            }
                        }
                    },
                    Err(e) => {
                        training_state.is_training = false;
                        next_training_state.set(AppTrainingState::Idle);
                        game_state.train_message = format!("Erreur lors de l'évaluation: {}", e);
                    }
                }
            },
            Err(e) => {
                training_state.is_training = false;
                next_training_state.set(AppTrainingState::Idle);
                game_state.train_message = format!("Erreur lors de l'entraînement: {}", e);
            }
        }

        // Remettre le modèle dans la ressource
        training_state.selected_model = Some(model);
    }
}