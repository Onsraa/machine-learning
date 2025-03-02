use crate::data::dataset_processor::GameImageDataset;
use bevy::prelude::*;
use nalgebra::DVector;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Resource)]
pub struct GameImageState {
    // Dataset
    pub dataset_path: Option<PathBuf>,
    pub dataset_loaded: bool,
    pub dataset: Option<Arc<GameImageDataset>>,

    // Traitement des images
    pub selected_image_path: Option<PathBuf>,
    pub processed_image: Option<DVector<f64>>,
    pub image_path_input: String,

    // Interface utilisateur
    pub dataset_folder_path: String,
    pub model_name: String,

    // Résultats de classification
    pub prediction_result: Option<(usize, Vec<f64>)>,

    // État d'entraînement
    pub train_progress: f32,
    pub train_message: String,
    pub train_epochs: usize,
    pub loss_history: Vec<(f64, f64)>,
    pub best_model_saved: bool,
}

impl Default for GameImageState {
    fn default() -> Self {
        Self {
            dataset_path: None,
            dataset_loaded: false,
            dataset: None,
            selected_image_path: None,
            processed_image: None,
            image_path_input: "predict_dataset/".to_string(),
            dataset_folder_path: "dataset".to_string(),
            model_name: "Le Saint Prédicteur".to_string(),
            prediction_result: None,
            train_progress: 0.0,
            train_message: "Ready for training".to_string(),
            train_epochs: 0,
            loss_history: Vec::new(),
            best_model_saved: false,
        }
    }
}