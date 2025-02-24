use crate::algorithms::linear_regression::LinearRegression;
use crate::algorithms::mlp::{Activation, MLP};
use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::{DataModel, DatasetConverter};
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;

pub fn training_system(
    mut training_state: ResMut<TrainingState>,
    dataset: Res<DatasetConverter>,
    data_model: Res<DataModel>,
    time: Res<Time>,
) {
    if training_state.should_reset {
        training_state.metrics.reset();
        training_state.selected_model = None;
        training_state.should_reset = false;
        println!("Training state has been reset");
        return;
    }

    if !training_state.is_training {
        return;
    }

    let current_time = time.elapsed_secs();
    if current_time - training_state.last_update < training_state.hyperparameters.epoch_interval {
        return;
    }
    training_state.last_update = current_time;

    println!("Training system running... Time: {:.2}s", current_time);

    if dataset.inputs.nrows() == 0 {
        println!("No data available for training!");
        return;
    }
    println!("Training with {} samples", dataset.inputs.nrows());

    let learning_rate = training_state.hyperparameters.learning_rate;
    let batch_size = training_state.hyperparameters.batch_size;
    let train_ratio = training_state.hyperparameters.train_ratio;

    if training_state.selected_model.is_none() {
        if data_model.is_classification() {
            println!("Initializing MLP for classification...");
            training_state.selected_model = Some(ModelAlgorithm::MLP(MLP::new(
                data_model.input_dim(),
                vec![5],
                data_model.n_classes().unwrap(),
                vec![Activation::Tanh, Activation::Tanh],
            )));
        } else {
            println!("Initializing LinearRegression for regression...");
            training_state.selected_model = Some(ModelAlgorithm::LinearRegression(
                LinearRegression::new(data_model.input_dim()),
            ));
        }
    }

    let n_samples = dataset.inputs.nrows();
    let n_train = (n_samples as f64 * train_ratio) as usize;
    println!("Using {} samples for training", n_train);

    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let (train_indices, test_indices) = indices.split_at(n_train);
    let batch_indices: Vec<usize> = train_indices
        .choose_multiple(&mut rng, batch_size.min(n_train))
        .cloned()
        .collect();

    let batch_inputs = select_rows(&dataset.inputs, &batch_indices);
    let batch_targets = select_elements(&dataset.outputs, &batch_indices);
    let test_inputs = select_rows(&dataset.inputs, test_indices);
    let test_targets = select_elements(&dataset.outputs, test_indices);

    if let Some(model) = training_state.selected_model.as_mut() {
        println!("Running training step...");
        let train_loss = model.fit(&batch_inputs, &batch_targets, learning_rate, 1);
        let test_loss = model.evaluate(&test_inputs, &test_targets);
        training_state.metrics.add_metrics(train_loss, test_loss);
        println!(
            "Epoch {}: Train Loss = {:.6}, Test Loss = {:.6}",
            training_state.metrics.current_epoch, train_loss, test_loss
        );
    }
}

fn select_rows(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), matrix.ncols(), |i, j| {
        matrix[(indices[i], j)]
    })
}

fn select_elements(vector: &DVector<f64>, indices: &[usize]) -> DMatrix<f64> {
    let data: Vec<f64> = indices.iter().map(|&i| vector[i]).collect();
    DMatrix::from_column_slice(indices.len(), 1, &data)
}
