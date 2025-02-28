use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::{DataModel, DatasetConverter};
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use std::time::Instant;

pub fn training_system(
    mut training_state: ResMut<TrainingState>,
    dataset: Res<DatasetConverter>,
    data_model: Res<DataModel>,
    time: Res<Time>,
) {
    // Gestion du reset
    if training_state.should_reset {
        training_state.metrics.reset();
        training_state.selected_model = None;
        training_state.should_reset = false;
        training_state.error_message = None;
        println!("Training state has been reset");
        return;
    }

    // Retour si on n'est pas en mode entraînement
    if !training_state.is_training {
        return;
    }

    // Gestion de l'intervalle entre les époques
    let current_time = time.elapsed_secs();
    if current_time - training_state.last_update < training_state.hyperparameters.epoch_interval {
        return;
    }
    training_state.last_update = current_time;

    // Vérification que les données ne sont pas vides
    if dataset.inputs.nrows() == 0 {
        training_state.error_message = Some("No data available for training!".to_string());
        println!("No data available for training!");
        return;
    }

    // Affichage des informations de début d'entraînement
    println!("\n--------------------------");
    println!("Training system running... Time: {:.2}s", current_time);
    println!("Training with {} samples", dataset.inputs.nrows());

    // Récupération des hyperparamètres
    let learning_rate = training_state.hyperparameters.learning_rate;
    let batch_size = training_state.hyperparameters.batch_size;
    let train_ratio = training_state.hyperparameters.train_ratio;
    let early_stopping_patience = training_state.hyperparameters.early_stopping_patience;

    // Initialisation du modèle si nécessaire
    if training_state.selected_model.is_none() {
        if data_model.is_classification() {
            println!("No model selected. Initializing LinearClassifier for classification...");
            training_state.selected_model = Some(ModelAlgorithm::new_linear_classifier(
                data_model.input_dim(),
                data_model.n_classes().unwrap_or(2),
            ));
        } else {
            println!("No model selected. Initializing LinearRegression for regression...");
            training_state.selected_model = Some(ModelAlgorithm::new_linear_regression(
                data_model.input_dim(),
            ));
        }
    }

    // Vérification que le modèle est correctement initialisé
    if training_state.selected_model.is_none() {
        training_state.error_message = Some("Failed to initialize model!".to_string());
        return;
    }

    // Nombre d'échantillons
    let n_samples = dataset.inputs.nrows();

    // Normalisation des données si nécessaire
    let (normalized_inputs, original_inputs) = if data_model.is_normalized() {
        // Données déjà normalisées dans le modèle
        (dataset.inputs.clone(), dataset.inputs.clone())
    } else {
        normalize_data(&dataset.inputs)
    };

    // *** GESTION DU TRAIN_RATIO, Y COMPRIS LE CAS 1.0 ***
    let (batch_inputs, batch_targets, test_inputs, test_targets) = if train_ratio >= 0.999 {
        // CAS SPÉCIAL: Utilisation de tous les échantillons pour l'entraînement ET l'évaluation
        println!("Using all {} samples for both training and evaluation (train_ratio=1.0)", n_samples);

        // Pour le batch, prendre soit tous les échantillons, soit un sous-ensemble selon le batch_size
        let effective_batch_size = batch_size.min(n_samples);

        // Si batch_size est inférieur au nombre d'échantillons, sélectionner aléatoirement
        let batch_indices = if effective_batch_size < n_samples {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rand::thread_rng());
            indices.iter().take(effective_batch_size).cloned().collect::<Vec<usize>>()
        } else {
            // Sinon, prendre tous les échantillons
            (0..n_samples).collect::<Vec<usize>>()
        };

        // Sélection des données pour le batch
        let batch_inputs = select_rows(&normalized_inputs, &batch_indices);
        let batch_targets = select_elements(&dataset.outputs, &batch_indices);

        // Pour l'évaluation, utiliser TOUS les échantillons (puisque train_ratio=1.0)
        let all_indices: Vec<usize> = (0..n_samples).collect();
        let test_inputs = select_rows(&normalized_inputs, &all_indices);
        let test_targets = select_elements(&dataset.outputs, &all_indices);

        (batch_inputs, batch_targets, test_inputs, test_targets)
    } else {
        // CAS NORMAL: Division train/test selon le ratio
        let n_train = (n_samples as f64 * train_ratio) as usize;
        if n_train == 0 {
            training_state.error_message = Some("Training set is empty! Adjust train ratio.".to_string());
            println!("Training set is empty!");
            return;
        }
        println!("Using {} samples for training, {} for testing", n_train, n_samples - n_train);

        // Division des données en ensembles d'entraînement et de test
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rand::thread_rng());
        let (train_indices, test_indices) = indices.split_at(n_train);

        // Vérification que le batch_size ne dépasse pas le nombre d'échantillons d'entraînement
        let effective_batch_size = batch_size.min(n_train);
        let batch_indices: Vec<usize> = train_indices
            .choose_multiple(&mut rand::thread_rng(), effective_batch_size)
            .cloned()
            .collect();

        // Préparation des données d'entraînement et de test
        let batch_inputs = select_rows(&normalized_inputs, &batch_indices);
        let batch_targets = select_elements(&dataset.outputs, &batch_indices);
        let test_inputs = select_rows(&normalized_inputs, test_indices);
        let test_targets = select_elements(&dataset.outputs, test_indices);

        (batch_inputs, batch_targets, test_inputs, test_targets)
    };

    // Vérification que les batchs ne sont pas vides
    if batch_inputs.nrows() == 0 || batch_targets.nrows() == 0 {
        training_state.error_message = Some("Empty batch! Check batch size.".to_string());
        println!("Empty batch! Check batch size.");
        return;
    }

    if test_inputs.nrows() == 0 || test_targets.nrows() == 0 {
        training_state.error_message = Some("Empty test set! Check train ratio.".to_string());
        println!("Empty test set! Check train ratio.");
        return;
    }

    // Affichage de statistiques sur les données
    println!("Input data range: min={:.3}, max={:.3}",
             normalized_inputs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             normalized_inputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    println!("Target data range: min={:.3}, max={:.3}",
             dataset.outputs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             dataset.outputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // Détermination du type de modèle pour les logs
    let model_type = match &training_state.selected_model {
        Some(ModelAlgorithm::LinearRegression(_, _)) => "Linear Regression",
        Some(ModelAlgorithm::LinearClassifier(_, _)) => "Linear Classifier",
        Some(ModelAlgorithm::MLP(_, _)) => {
            if training_state.selected_model.as_ref().unwrap().is_classification() {
                "MLP (Classification)"
            } else {
                "MLP (Regression)"
            }
        },
        Some(ModelAlgorithm::RBF(_, _)) => {
            if training_state.selected_model.as_ref().unwrap().is_classification() {
                "RBF (Classification)"
            } else {
                "RBF (Regression)"
            }
        },
        None => "Unknown",
    };

    println!("Running training step for {} model", model_type);

    // Prise de possession temporaire du modèle pour éviter les conflits d'emprunt
    if let Some(mut model) = training_state.selected_model.take() {
        // Entraînement du modèle et mesure du temps
        let start_time = Instant::now();
        match model.fit(&batch_inputs, &batch_targets, learning_rate, 1) {
            Ok(train_loss) => {
                let fit_time = start_time.elapsed();

                // Évaluation sur les données de test
                match model.evaluate(&test_inputs, &test_targets) {
                    Ok(test_loss) => {
                        // Mise à jour des métriques
                        training_state.metrics.add_metrics(train_loss, test_loss);
                        println!(
                            "Epoch {}: Train Loss = {:.6}, Test Loss = {:.6} (Fit time: {:?})",
                            training_state.metrics.current_epoch, train_loss, test_loss, fit_time
                        );

                        // Affichage de prédictions pour quelques échantillons de test
                        println!("Sample predictions:");
                        for i in 0..test_inputs.nrows() {
                            let input = DVector::from_iterator(
                                test_inputs.ncols(),
                                test_inputs.row(i).iter().cloned()
                            );
                            if let Ok(pred) = model.predict(&input) {
                                println!("  Sample {}: Prediction = {:.3}, Actual = {:.3}",
                                         i, pred[0], test_targets[(i, 0)]);
                            }
                        }

                        // Suppression des erreurs précédentes
                        training_state.error_message = None;

                        // Vérification de l'early stopping
                        // if training_state.metrics.should_stop_early(early_stopping_patience) {
                        //     println!("Early stopping triggered - no improvement for {} epochs",
                        //              early_stopping_patience);
                        //     training_state.is_training = false;
                        // }
                    },
                    Err(e) => {
                        training_state.error_message = Some(format!("Error evaluating model: {}", e));
                        println!("Error evaluating model: {}", e);
                    },
                }
            },
            Err(e) => {
                training_state.error_message = Some(format!("Error training model: {}", e));
                println!("Error training model: {}", e);
            },
        }

        // Remise du modèle à sa place
        training_state.selected_model = Some(model);
    }
    println!("--------------------------\n");
}

// Fonction utilitaire pour normaliser les données (soustraire la moyenne, diviser par l'écart-type)
fn normalize_data(data: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut normalized = data.clone();
    let original = data.clone();

    for j in 0..data.ncols() {
        let column_values: Vec<f64> = (0..data.nrows()).map(|i| data[(i, j)]).collect();

        // Calcul de la moyenne et de l'écart-type
        let mean = column_values.iter().sum::<f64>() / column_values.len() as f64;
        let variance = column_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / column_values.len() as f64;
        let std_dev = variance.sqrt().max(1e-8); // Éviter la division par zéro

        // Normalisation de la colonne
        for i in 0..data.nrows() {
            normalized[(i, j)] = (data[(i, j)] - mean) / std_dev;
        }
    }

    (normalized, original)
}

// Fonction utilitaire pour sélectionner des lignes d'une matrice
fn select_rows(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    if indices.is_empty() {
        return DMatrix::zeros(0, matrix.ncols());
    }
    DMatrix::from_fn(indices.len(), matrix.ncols(), |i, j| {
        matrix[(indices[i], j)]
    })
}

// Fonction utilitaire pour sélectionner des éléments d'un vecteur
fn select_elements(vector: &DVector<f64>, indices: &[usize]) -> DMatrix<f64> {
    if indices.is_empty() {
        return DMatrix::zeros(0, 1);
    }
    let data: Vec<f64> = indices.iter().map(|&i| vector[i]).collect();
    DMatrix::from_column_slice(indices.len(), 1, &data)
}

// Fonction utilitaire pour min
fn min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}