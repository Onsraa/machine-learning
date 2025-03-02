use crate::algorithms::model_selector::ModelAlgorithm;
use crate::data::universal_dataset::TaskType;
use bevy::prelude::*;
use chrono::Local;
use ron::ser::{to_string_pretty, PrettyConfig};
use serde::{Deserialize, Serialize};
use std::fs::{self, create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Resource)]
pub struct ModelManager {
    pub base_directory: PathBuf,
    pub save_directory: PathBuf,
    pub model_infos: Vec<ModelSaveInfo>,
    pub selected_model_index: Option<usize>,
    pub save_dialog_open: bool,
    pub dialog_model_name: String,
    pub dialog_description: String,
    pub status_message: Option<(String, f32)>,
    pub confirm_delete_dialog_open: bool,
    pub model_to_delete: Option<usize>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self {
            base_directory: "saved_models".parse().unwrap(),
            save_directory: Default::default(),
            model_infos: vec![],
            selected_model_index: None,
            save_dialog_open: false,
            dialog_model_name: "".to_string(),
            dialog_description: "".to_string(),
            status_message: None,
            confirm_delete_dialog_open: false,
            model_to_delete: None,
        }
    }
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
    pub category: String,
}

impl ModelManager {
    pub fn new() -> Self {
        let base_dir = PathBuf::from("saved_models");

        // Créer les sous-répertoires
        let cas_tests_dir = base_dir.join("cas_de_tests");
        let images_jeux_dir = base_dir.join("images_jeux");

        for dir in &[&base_dir, &cas_tests_dir, &images_jeux_dir] {
            if !dir.exists() {
                if let Err(e) = create_dir_all(dir) {
                    eprintln!("Erreur lors de la création du répertoire {}: {}", dir.display(), e);
                }
            }
        }

        let mut model_infos = Vec::new();
        model_infos.extend(Self::load_model_infos_from_dir(&cas_tests_dir, "cas_de_tests"));
        model_infos.extend(Self::load_model_infos_from_dir(&images_jeux_dir, "images_jeux"));

        Self {
            base_directory: base_dir.clone(),
            save_directory: base_dir,
            model_infos,
            selected_model_index: None,
            save_dialog_open: false,
            dialog_model_name: String::new(),
            dialog_description: String::new(),
            status_message: None,
            confirm_delete_dialog_open: false,
            model_to_delete: None,
        }
    }

    fn load_model_infos_from_dir(dir: &Path, category: &str) -> Vec<ModelSaveInfo> {
        let mut infos = Vec::new();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "info") {
                    if let Ok(mut file) = File::open(&path) {
                        let mut contents = String::new();
                        if file.read_to_string(&mut contents).is_ok() {
                            if let Ok(mut info) = ron::from_str::<ModelSaveInfo>(&contents) {
                                info.category = category.to_string();
                                infos.push(info);
                            }
                        }
                    }
                }
            }
        }

        infos
    }

    pub fn set_save_category(&mut self, category: &str) {
        self.save_directory = self.base_directory.join(category);

        if !self.save_directory.exists() {
            if let Err(e) = create_dir_all(&self.save_directory) {
                eprintln!("Erreur lors de la création du répertoire {}: {}", self.save_directory.display(), e);
            }
        }
    }

    pub fn save_model_with_category(
        &mut self,
        model: &ModelAlgorithm,
        name: &str,
        description: Option<String>,
        category: &str,
    ) -> Result<(), String> {
        self.set_save_category(category);

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

        let model_file_path = self.save_directory.join(format!("{}.model", base_file_name));
        let info_file_path = self.save_directory.join(format!("{}.info", base_file_name));

        let model_data = match ron::to_string(model) {
            Ok(data) => data,
            Err(e) => return Err(format!("Erreur lors de la sérialisation du modèle: {}", e)),
        };

        if let Err(e) = File::create(&model_file_path)
            .and_then(|mut file| file.write_all(model_data.as_bytes()))
        {
            return Err(format!("Erreur lors de la sauvegarde du fichier modèle: {}", e));
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
            category: category.to_string(),
        };

        let pretty_config = PrettyConfig::default();

        let info_data = match to_string_pretty(&info, pretty_config) {
            Ok(data) => data,
            Err(e) => return Err(format!("Erreur lors de la sérialisation des informations du modèle: {}", e)),
        };

        if let Err(e) = File::create(&info_file_path)
            .and_then(|mut file| file.write_all(info_data.as_bytes()))
        {
            return Err(format!("Erreur lors de la sauvegarde du fichier d'informations: {}", e));
        }

        self.model_infos.push(info);
        self.set_status(format!("Modèle {} sauvegardé avec succès", name), 3.0);

        Ok(())
    }

    pub fn save_model(
        &mut self,
        model: &ModelAlgorithm,
        name: &str,
        description: Option<String>,
    ) -> Result<(), String> {
        self.save_model_with_category(model, name, description, "cas_de_tests")
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

        println!("Chargement du modèle: {}", path.display());

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => return Err(format!("Failed to open model file: {}", e)),
        };

        let mut buffer = String::new();
        if let Err(e) = file.read_to_string(&mut buffer) {
            return Err(format!("Failed to read model file: {}", e));
        }

        println!("Contenu lu, longueur: {} octets", buffer.len());

        match ron::from_str::<ModelAlgorithm>(&buffer) {
            Ok(model) => {
                println!("Modèle chargé avec succès (format RON)");
                Ok(model)
            },
            Err(ron_err) => {
                println!("Erreur de désérialisation RON: {}", ron_err);

                let file = match File::open(path) {
                    Ok(file) => file,
                    Err(e) => return Err(format!("Failed to reopen model file: {}", e)),
                };

                match bincode::deserialize_from::<_, ModelAlgorithm>(file) {
                    Ok(model) => {
                        println!("Modèle chargé avec succès (format Bincode)");
                        Ok(model)
                    },
                    Err(bin_err) => Err(format!("Failed to deserialize model: RON error: {}, Bincode error: {}", ron_err, bin_err)),
                }
            }
        }
    }

    pub fn delete_model(&mut self, index: usize) -> Result<(), String> {
        if index >= self.model_infos.len() {
            return Err("Invalid model index".to_string());
        }

        let info = &self.model_infos[index];
        let model_path = Path::new(&info.file_path);
        let info_path = model_path.with_extension("info");

        if model_path.exists() {
            if let Err(e) = fs::remove_file(model_path) {
                println!("Échec de la suppression du fichier modèle: {}", e);
            }
        }

        if info_path.exists() {
            if let Err(e) = fs::remove_file(info_path) {
                println!("Échec de la suppression du fichier info: {}", e);
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

        println!("Modèle {} supprimé avec succès", model_name);
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
            Err("Aucun modèle sélectionné pour suppression".to_string())
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

    pub fn refresh(&mut self) {
        let cas_tests_dir = self.base_directory.join("cas_de_tests");
        let images_jeux_dir = self.base_directory.join("images_jeux");

        self.model_infos.clear();
        self.model_infos.extend(Self::load_model_infos_from_dir(&cas_tests_dir, "cas_de_tests"));
        self.model_infos.extend(Self::load_model_infos_from_dir(&images_jeux_dir, "images_jeux"));

        self.selected_model_index = None;
    }
}