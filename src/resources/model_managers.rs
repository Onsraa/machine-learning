use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::universal_dataset::TaskType;
use bevy::prelude::*;
use chrono::{DateTime, Local};
use ron::de::from_str;
use ron::ser::{to_string_pretty, PrettyConfig};
use serde::{Deserialize, Serialize};
use std::fs::{self, create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Resource, Default)]
pub struct ModelManager {
    pub save_directory: PathBuf,
    pub model_infos: Vec<ModelSaveInfo>,
    pub selected_model_index: Option<usize>,
    pub save_dialog_open: bool,
    pub load_dialog_open: bool,
    pub dialog_model_name: String,
    pub dialog_description: String,
    pub status_message: Option<(String, f32)>,
    pub confirm_delete_dialog_open: bool,
    pub model_to_delete: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelSaveInfo {
    pub name: String,
    pub model_type: String,
    pub task_type: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub accuracy: Option<f64>,
    pub created_at: String,
    pub description: Option<String>,
    pub file_path: String,
}

impl ModelManager {
    pub fn new() -> Self {
        let save_dir = PathBuf::from("saved_models");
        if !save_dir.exists() {
            if let Err(e) = create_dir_all(&save_dir) {
                eprintln!("Failed to create save directory: {}", e);
            }
        }
        let model_infos = Self::load_model_infos(&save_dir);
        Self {
            save_directory: save_dir,
            model_infos,
            selected_model_index: None,
            save_dialog_open: false,
            load_dialog_open: false,
            dialog_model_name: String::new(),
            dialog_description: String::new(),
            status_message: None,
            confirm_delete_dialog_open: false,
            model_to_delete: None,
        }
    }

    fn load_model_infos(save_dir: &Path) -> Vec<ModelSaveInfo> {
        let mut infos = Vec::new();

        if let Ok(entries) = fs::read_dir(save_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "info") {
                    if let Ok(mut file) = File::open(&path) {
                        let mut contents = String::new();
                        if file.read_to_string(&mut contents).is_ok() {
                            if let Ok(info) = from_str::<ModelSaveInfo>(&contents) {
                                infos.push(info);
                            }
                        }
                    }
                }
            }
        }

        infos
    }

    pub fn save_model(
        &mut self,
        model: &ModelAlgorithm,
        name: &str,
        description: Option<String>,
    ) -> Result<(), String> {
        let model_type = match model {
            ModelAlgorithm::LinearRegression(_, _) => "LinearRegression",
            ModelAlgorithm::LinearClassifier(_, _) => "LinearClassifier",
            ModelAlgorithm::MLP(_, _) => "MLP",
            ModelAlgorithm::RBF(_, _) => "RBF",
            ModelAlgorithm::SVM(_, _) => "SVM",
        };

        let task_type = match model.get_task_type() {
            TaskType::Classification => "Classification",
            TaskType::Regression => "Regression",
        };

        let base_file_name = format!(
            "{}_{}_{}",
            name.replace(" ", "_"),
            model_type,
            Local::now().format("%Y%m%d_%H%M%S")
        );

        let model_file_path = self
            .save_directory
            .join(format!("{}.model", base_file_name));
        let info_file_path = self.save_directory.join(format!("{}.info", base_file_name));

        let model_data = match ron::to_string(model) {
            Ok(data) => data,
            Err(e) => return Err(format!("Failed to serialize model: {}", e)),
        };

        if let Err(e) = File::create(&model_file_path)
            .and_then(|mut file| file.write_all(model_data.as_bytes()))
        {
            return Err(format!("Failed to save model file: {}", e));
        }

        let info = ModelSaveInfo {
            name: name.to_string(),
            model_type: model_type.to_string(),
            task_type: task_type.to_string(),
            input_dim: match model {
                ModelAlgorithm::LinearRegression(m, _) => m.weights.len(),
                ModelAlgorithm::LinearClassifier(m, _) => {
                    if !m.classifiers.is_empty() {
                        m.classifiers[0].len()
                    } else {
                        0
                    }
                }
                ModelAlgorithm::MLP(m, _) => {
                    if !m.layers.is_empty() {
                        m.layers[0].weights.ncols()
                    } else {
                        0
                    }
                }
                ModelAlgorithm::RBF(m, _) => m.centers.ncols(),
                ModelAlgorithm::SVM(m, _) => m.input_dim,
            },
            output_dim: match model {
                ModelAlgorithm::LinearRegression(_, _) => 1,
                ModelAlgorithm::LinearClassifier(m, _) => m.n_classes,
                ModelAlgorithm::MLP(m, _) => {
                    if !m.layers.is_empty() {
                        m.layers.last().unwrap().weights.nrows()
                    } else {
                        0
                    }
                }
                ModelAlgorithm::RBF(_, _) => 1,
                ModelAlgorithm::SVM(_, _) => 1,
            },
            accuracy: None,
            created_at: Local::now().to_rfc3339(),
            description,
            file_path: model_file_path.to_string_lossy().to_string(),
        };

        let pretty_config = PrettyConfig::default();

        let info_data = match to_string_pretty(&info, pretty_config) {
            Ok(data) => data,
            Err(e) => return Err(format!("Failed to serialize model info: {}", e)),
        };

        if let Err(e) =
            File::create(&info_file_path).and_then(|mut file| file.write_all(info_data.as_bytes()))
        {
            return Err(format!("Failed to save model info file: {}", e));
        }

        self.model_infos.push(info);
        self.set_status(format!("Model {} saved successfully", name), 3.0);

        Ok(())
    }

    pub fn load_model(&self, index: usize) -> Result<ModelAlgorithm, String> {
        if index >= self.model_infos.len() {
            return Err("Invalid model index".to_string());
        }

        let info = &self.model_infos[index];
        let path = Path::new(&info.file_path);

        if !path.exists() {
            return Err(format!("Model file not found: {}", path.display()));
        }

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => return Err(format!("Failed to open model file: {}", e)),
        };

        let mut buffer = String::new();
        if let Err(e) = file.read_to_string(&mut buffer) {
            return Err(format!("Failed to read model file: {}", e));
        }

        match ron::from_str::<ModelAlgorithm>(&buffer) {
            Ok(model) => Ok(model),
            Err(e) => Err(format!("Failed to deserialize model: {}", e)),
        }
    }

    pub fn delete_model(&mut self, index: usize) -> Result<(), String> {
        if index >= self.model_infos.len() {
            return Err("Invalid model index".to_string());
        }

        let info = &self.model_infos[index];
        let model_path = Path::new(&info.file_path);
        let info_path = model_path.with_extension("info");

        // Supprimer les fichiers
        if model_path.exists() {
            if let Err(e) = fs::remove_file(model_path) {
                return Err(format!("Failed to delete model file: {}", e));
            }
        }

        if info_path.exists() {
            if let Err(e) = fs::remove_file(info_path) {
                return Err(format!("Failed to delete info file: {}", e));
            }
        }

        let model_name = self.model_infos[index].name.clone();
        self.model_infos.remove(index);

        if let Some(selected) = self.selected_model_index {
            if selected == index {
                self.selected_model_index = None;
            } else if selected > index {
                self.selected_model_index = Some(selected - 1);
            }
        }

        self.set_status(format!("Model {} deleted", model_name), 3.0);

        Ok(())
    }

    pub fn request_delete_confirmation(&mut self, index: usize) {
        self.model_to_delete = Some(index);
        self.confirm_delete_dialog_open = true;
    }

    pub fn confirm_delete(&mut self) -> Result<(), String> {
        if let Some(index) = self.model_to_delete.take() {
            let result = self.delete_model(index);
            self.confirm_delete_dialog_open = false;
            result
        } else {
            Err("No model selected for deletion".to_string())
        }
    }

    pub fn cancel_delete(&mut self) {
        self.model_to_delete = None;
        self.confirm_delete_dialog_open = false;
    }

    pub fn set_status(&mut self, message: String, duration: f32) {
        self.status_message = Some((message, duration));
    }

    pub fn update_status(&mut self, delta_time: f32) {
        if let Some((_, ref mut duration)) = self.status_message {
            *duration -= delta_time;
            if *duration <= 0.0 {
                self.status_message = None;
            }
        }
    }
}
