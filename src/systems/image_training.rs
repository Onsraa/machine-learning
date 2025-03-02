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
    if *app_training_state.get() != AppTrainingState::Training || !training_state.is_training {
        return;
    }

    if game_state.dataset.is_none() || training_state.selected_model.is_none() {
        next_training_state.set(AppTrainingState::Idle);
        training_state.is_training = false;
        game_state.train_message = "Error: Dataset or model not defined".to_string();
        return;
    }

    if let Some(ref dataset) = game_state.dataset {
        if !dataset.data.is_empty() {
            let sample_vec_size = dataset.data[0].len();
            // println!("Taille du vecteur d'entrée des échantillons: {}", sample_vec_size);

            // Vérifier que le modèle a la bonne dimension d'entrée
            if let Some(ref model) = training_state.selected_model {
                if let crate::algorithms::model_selector::ModelAlgorithm::MLP(ref mlp, _) = model {
                    if !mlp.layers.is_empty() {
                        let model_input_size = mlp.layers[0].weights.ncols();
                        // println!("Taille d'entrée du modèle: {}", model_input_size);

                        if model_input_size != sample_vec_size {
                            next_training_state.set(AppTrainingState::Idle);
                            training_state.is_training = false;
                            game_state.train_message = format!(
                                "Incompatibilité de dimension: modèle({}) != données({})",
                                model_input_size, sample_vec_size
                            );
                            return;
                        }
                    }
                }
            }
        }
    }

    let current_time = time.elapsed_secs();
    if current_time - training_state.last_update < training_state.hyperparameters.epoch_interval {
        return;
    }
    training_state.last_update = current_time;

    game_state.train_epochs += 1;

    let train_ratio = training_state.hyperparameters.train_ratio;
    let batch_size = training_state.hyperparameters.batch_size;
    let learning_rate = training_state.hyperparameters.learning_rate;

    let num_classes;
    let dataset_size;

    // Prepare the data for training
    let batch_inputs;
    let batch_targets;
    let test_inputs;
    let test_targets;

    {
        let dataset = game_state.dataset.as_ref().unwrap();
        num_classes = dataset.category_mapping.len();
        dataset_size = dataset.data.len();

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        if training_state.metrics.training_losses.is_empty() {
            let (new_train_idx, new_test_idx) = DatasetProcessor::split_dataset(dataset, train_ratio);
            train_indices = new_train_idx;
            test_indices = new_test_idx;

            training_state.index_cache = Some((train_indices.clone(), test_indices.clone()));
        } else {
            if let Some(ref idx_cache) = training_state.index_cache {
                train_indices = idx_cache.0.clone();
                test_indices = idx_cache.1.clone();
            } else {
                let (new_train_idx, new_test_idx) = DatasetProcessor::split_dataset(dataset, train_ratio);
                train_indices = new_train_idx;
                test_indices = new_test_idx;

                training_state.index_cache = Some((train_indices.clone(), test_indices.clone()));
            }
        }

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

        let matrices = DatasetProcessor::dataset_to_matrices(dataset, &batch_indices);
        batch_inputs = matrices.0;
        batch_targets = matrices.1;

        let test_matrices = DatasetProcessor::dataset_to_matrices(dataset, &test_indices);
        test_inputs = test_matrices.0;
        test_targets = test_matrices.1;
    }

    if let Some(mut model) = training_state.selected_model.take() {
        match model.fit(&batch_inputs, &batch_targets, learning_rate, 1) {
            Ok(train_loss) => {
                match model.evaluate(&test_inputs, &test_targets) {
                    Ok(test_loss) => {
                        training_state.metrics.add_metrics(train_loss, test_loss);

                        game_state.loss_history.push((train_loss, test_loss));

                        game_state.train_message = format!(
                            "Epoch {}: Train Loss = {:.6}, Test Loss = {:.6}",
                            game_state.train_epochs,
                            train_loss,
                            test_loss
                        );

                        let convergence = training_state.metrics.get_convergence_estimate() as f32;
                        game_state.train_progress = convergence;

                        if test_loss <= training_state.metrics.best_test_loss && !game_state.best_model_saved {
                            // game_state.best_model_saved = true;
                            game_state.train_message = format!(
                                "{} - New best model (loss: {:.6})!",
                                game_state.train_message,
                                test_loss
                            );
                        }
                    },
                    Err(e) => {
                        training_state.is_training = false;
                        next_training_state.set(AppTrainingState::Idle);
                        game_state.train_message = format!("Error during evaluation: {}", e);
                    }
                }
            },
            Err(e) => {
                training_state.is_training = false;
                next_training_state.set(AppTrainingState::Idle);
                game_state.train_message = format!("Error during training: {}", e);
            }
        }
        training_state.selected_model = Some(model);
    }
}
