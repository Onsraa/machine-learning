use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;

use crate::algorithms::linear_classifier::LinearClassifier;
use crate::algorithms::linear_regression::LinearRegression;
use crate::data::{DataModel, DatasetConverter};
use crate::resources::training::TrainingState;

pub fn training_system(
    mut training_state: ResMut<TrainingState>,
    dataset: Res<DatasetConverter>,
    data_model: Res<DataModel>,
) {
    // Gérer le reset si nécessaire
    if training_state.should_reset {
        // Reset des métriques
        training_state.metrics.reset();

        // Reset des modèles
        training_state.regression_model = None;
        training_state.classification_model = None;

        // Reset du flag
        training_state.should_reset = false;
        println!("Training state has been reset");
        return;
    }

    // Vérification de l'état d'entraînement
    if !training_state.is_training {
        return;
    }

    println!("Training system running..."); // Debug

    // Vérifier si des données sont disponibles
    if dataset.inputs.nrows() == 0 {
        println!("No data available for training!");
        return;
    }
    println!("Training with {} samples", dataset.inputs.nrows());

    // Extraire les hyperparamètres
    let learning_rate = training_state.hyperparameters.learning_rate;
    let batch_size = training_state.hyperparameters.batch_size;
    let train_ratio = training_state.hyperparameters.train_ratio;

    // Initialiser le modèle si nécessaire
    if data_model.is_classification() {
        if training_state.classification_model.is_none() {
            println!("Initializing classification model...");
            training_state.classification_model = Some(LinearClassifier::new(
                data_model.input_dim(),
                data_model.n_classes().unwrap()
            ));
        }
    } else {
        if training_state.regression_model.is_none() {
            println!("Initializing regression model...");
            training_state.regression_model = Some(LinearRegression::new(
                data_model.input_dim()
            ));
        }
    }

    // Préparation des données d'entraînement
    let n_samples = dataset.inputs.nrows();
    let n_train = (n_samples as f64 * train_ratio) as usize;
    println!("Using {} samples for training", n_train);

    // Création des indices aléatoires
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    // Séparation train/test
    let (train_indices, test_indices) = indices.split_at(n_train);
    let batch_indices: Vec<usize> = train_indices
        .choose_multiple(&mut rng, batch_size.min(n_train))
        .cloned()
        .collect();

    // Préparation des données
    let batch_inputs = select_rows(&dataset.inputs, &batch_indices);
    let batch_outputs = select_elements(&dataset.outputs, &batch_indices);
    let test_inputs = select_rows(&dataset.inputs, test_indices);
    let test_outputs = select_elements(&dataset.outputs, test_indices);

    // Entraînement selon le type de modèle
    if data_model.is_classification() {
        if let Some(model) = &mut training_state.classification_model {
            println!("Running classification training step...");

            let train_loss = train_classification_batch(
                model,
                &batch_inputs,
                &batch_outputs,
                learning_rate
            );

            let test_loss = evaluate_classification(
                model,
                &test_inputs,
                &test_outputs
            );

            training_state.metrics.add_metrics(train_loss, test_loss);

            println!("Epoch {}: Train Loss = {:.6}, Test Loss = {:.6}",
                     training_state.metrics.current_epoch,
                     train_loss,
                     test_loss
            );
        }
    } else {
        if let Some(model) = &mut training_state.regression_model {
            println!("Running regression training step...");

            let train_loss = train_regression_batch(
                model,
                &batch_inputs,
                &batch_outputs,
                learning_rate
            );

            let test_loss = evaluate_regression(
                model,
                &test_inputs,
                &test_outputs
            );

            training_state.metrics.add_metrics(train_loss, test_loss);

            println!("Epoch {}: Train Loss = {:.6}, Test Loss = {:.6}",
                     training_state.metrics.current_epoch,
                     train_loss,
                     test_loss
            );
        }
    }
}

// Fonctions utilitaires pour la sélection des données
fn select_rows(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), matrix.ncols(), |i, j| {
        matrix[(indices[i], j)]
    })
}

fn select_elements(vector: &DVector<f64>, indices: &[usize]) -> DVector<f64> {
    DVector::from_iterator(indices.len(), indices.iter().map(|&i| vector[i]))
}

// Fonctions d'entraînement et d'évaluation pour la classification
fn train_classification_batch(
    model: &mut LinearClassifier,
    inputs: &DMatrix<f64>,
    outputs: &DVector<f64>,
    learning_rate: f64
) -> f64 {
    // Une seule itération de mini-batch
    let mut losses = model.fit(inputs, &outputs.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                               learning_rate, 1);
    losses.pop().unwrap_or(0.0)
}

fn evaluate_classification(
    model: &LinearClassifier,
    inputs: &DMatrix<f64>,
    outputs: &DVector<f64>
) -> f64 {
    let y_true: Vec<usize> = outputs.iter().map(|&x| x as usize).collect();
    1.0 - model.evaluate(inputs, &y_true)  // Convertit accuracy en loss
}

// Fonctions d'entraînement et d'évaluation pour la régression
fn train_regression_batch(
    model: &mut LinearRegression,
    inputs: &DMatrix<f64>,
    outputs: &DVector<f64>,
    learning_rate: f64
) -> f64 {
    // Une seule itération de mini-batch
    let mut losses = model.fit(inputs, outputs, learning_rate, 1);
    losses.pop().unwrap_or(0.0)
}

fn evaluate_regression(
    model: &LinearRegression,
    inputs: &DMatrix<f64>,
    outputs: &DVector<f64>
) -> f64 {
    model.evaluate(inputs, outputs)
}