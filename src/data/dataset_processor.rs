use image::{DynamicImage, GenericImageView, GrayImage};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use rand::seq::SliceRandom;

// Taille standard pour toutes les images
pub const TARGET_SIZE: (u32, u32) = (64, 64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameImageDataset {
    pub data: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub category_mapping: HashMap<String, usize>,
    pub reverse_mapping: HashMap<usize, String>,
    pub sample_paths: Vec<PathBuf>, // Chemins des images pour référence
}

impl Default for GameImageDataset {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            labels: Vec::new(),
            category_mapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
            sample_paths: Vec::new(),
        }
    }
}

pub struct DatasetProcessor;

impl DatasetProcessor {
    // Charge et traite le dataset complet depuis un dossier de base
    pub fn process_dataset<P: AsRef<Path>>(
        root_dir: P,
        save_path: Option<P>,
    ) -> Result<GameImageDataset, String> {
        let root = root_dir.as_ref();

        // Vérifier que le dossier existe
        if !root.exists() || !root.is_dir() {
            return Err(format!("Le dossier {} n'existe pas", root.display()));
        }

        // Trouver tous les sous-dossiers (catégories)
        let categories: Vec<String> = match fs::read_dir(root) {
            Ok(entries) => {
                entries
                    .filter_map(|entry| {
                        let entry = entry.ok()?;
                        let path = entry.path();
                        if path.is_dir() {
                            path.file_name()?.to_str().map(String::from)
                        } else {
                            None
                        }
                    })
                    .collect()
            },
            Err(e) => return Err(format!("Erreur lors de la lecture du dossier: {}", e)),
        };

        if categories.is_empty() {
            return Err("Aucune catégorie trouvée dans le dossier".to_string());
        }

        println!("Catégories trouvées: {:?}", categories);

        // Créer les mappings catégorie <-> index
        let mut category_mapping: HashMap<String, usize> = HashMap::new();
        let mut reverse_mapping: HashMap<usize, String> = HashMap::new();

        for (i, cat) in categories.iter().enumerate() {
            category_mapping.insert(cat.clone(), i);
            reverse_mapping.insert(i, cat.clone());
        }

        let mut data = Vec::new();
        let mut labels = Vec::new();
        let mut sample_paths = Vec::new();

        // Traiter chaque catégorie
        for (cat_name, cat_index) in &category_mapping {
            let cat_dir = root.join(cat_name);

            // Trouver toutes les images dans cette catégorie
            let mut image_paths: Vec<PathBuf> = Vec::new();

            // Extensions d'images supportées
            for ext in &["jpg", "jpeg", "png"] {
                let pattern = format!("{}/*.{}", cat_dir.display(), ext);
                match glob::glob(&pattern) {
                    Ok(paths) => {
                        for path in paths {
                            if let Ok(path) = path {
                                image_paths.push(path);
                            }
                        }
                    },
                    Err(e) => {
                        println!("Erreur de pattern glob {}: {}", pattern, e);
                    }
                }
            }

            println!("Trouvé {} images dans la catégorie {}", image_paths.len(), cat_name);

            // Si aucune image n'est trouvée, passer à la catégorie suivante
            if image_paths.is_empty() {
                continue;
            }

            // Traiter chaque image
            for img_path in image_paths {
                match Self::process_image(&img_path) {
                    Ok(img_vec) => {
                        data.push(img_vec.as_slice().to_vec());
                        labels.push(*cat_index);
                        sample_paths.push(img_path);
                    },
                    Err(e) => {
                        println!("Erreur lors du traitement de {}: {}", img_path.display(), e);
                    }
                }
            }
        }

        if data.is_empty() {
            return Err("Aucune image n'a pu être traitée".to_string());
        }

        let dataset = GameImageDataset {
            data,
            labels,
            category_mapping,
            reverse_mapping,
            sample_paths,
        };

        // Sauvegarder le dataset si un chemin est fourni
        if let Some(save_path) = save_path {
            match File::create(save_path.as_ref()) {
                Ok(file) => {
                    let writer = BufWriter::new(file);
                    match serde_json::to_writer(writer, &dataset) {
                        Ok(_) => println!("Dataset sauvegardé avec {} images", dataset.data.len()),
                        Err(e) => println!("Erreur lors de la sauvegarde du dataset: {}", e),
                    }
                },
                Err(e) => println!("Erreur lors de la création du fichier: {}", e),
            }
        }

        Ok(dataset)
    }

    // Traite une seule image
    pub fn process_image<P: AsRef<Path>>(path: P) -> Result<DVector<f64>, String> {
        // Charger l'image
        let img = match image::open(path.as_ref()) {
            Ok(img) => img,
            Err(e) => return Err(format!("Erreur lors du chargement de l'image: {}", e)),
        };

        // Convertir en niveaux de gris et redimensionner
        let processed = Self::preprocess_image(&img);

        // Convertir en vecteur
        Ok(Self::image_to_vector(&processed))
    }

    // Prétraite une image: redimensionnement et conversion en niveaux de gris
    pub fn preprocess_image(img: &DynamicImage) -> GrayImage {
        let gray_img = img.to_luma8();
        image::imageops::resize(
            &gray_img,
            TARGET_SIZE.0,
            TARGET_SIZE.1,
            image::imageops::FilterType::Lanczos3
        )
    }

    // Convertit une image en niveaux de gris en vecteur normalisé
    pub fn image_to_vector(img: &GrayImage) -> DVector<f64> {
        let flat_vec: Vec<f64> = img
            .pixels()
            .map(|p| p[0] as f64 / 255.0)  // Normalisation entre 0 et 1
            .collect();

        DVector::from_vec(flat_vec)
    }

    // Divise le dataset en ensembles d'entraînement et de test
    pub fn split_dataset(
        dataset: &GameImageDataset,
        train_ratio: f64
    ) -> (Vec<usize>, Vec<usize>) {
        let n_samples = dataset.data.len();
        let n_train = (n_samples as f64 * train_ratio) as usize;

        // Mélanger les indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        // Diviser en ensembles d'entraînement et de test
        let train_indices = indices[..n_train].to_vec();
        let test_indices = indices[n_train..].to_vec();

        (train_indices, test_indices)
    }

    // Convertit les données du dataset en matrices pour l'entraînement
    pub fn dataset_to_matrices(
        dataset: &GameImageDataset,
        indices: &[usize]
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let n_samples = indices.len();
        if n_samples == 0 {
            return (DMatrix::zeros(0, 0), DMatrix::zeros(0, 0));
        }

        let n_features = dataset.data[0].len();

        // Créer les matrices
        let mut x_matrix = DMatrix::zeros(n_samples, n_features);
        let mut y_matrix = DMatrix::zeros(n_samples, 1);

        // Remplir les matrices
        for (i, &idx) in indices.iter().enumerate() {
            // Copier les features
            for j in 0..n_features {
                x_matrix[(i, j)] = dataset.data[idx][j];
            }

            // Étiquette de classe
            y_matrix[(i, 0)] = dataset.labels[idx] as f64;
        }

        (x_matrix, y_matrix)
    }

    // Charge un dataset depuis un fichier
    pub fn load_dataset<P: AsRef<Path>>(path: P) -> Result<GameImageDataset, String> {
        match File::open(path.as_ref()) {
            Ok(file) => {
                match serde_json::from_reader(file) {
                    Ok(dataset) => Ok(dataset),
                    Err(e) => Err(format!("Erreur lors de la désérialisation du dataset: {}", e)),
                }
            },
            Err(e) => Err(format!("Erreur lors de l'ouverture du fichier: {}", e)),
        }
    }
}