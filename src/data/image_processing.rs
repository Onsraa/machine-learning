use image::{DynamicImage};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

pub const TARGET_SIZE: (u32, u32) = (128, 72);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDataset {
    pub data: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub category_mapping: HashMap<String, usize>,
    pub reverse_mapping: HashMap<usize, String>,
}

pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// Redimensionne une image et la convertit en niveau de gris
    pub fn preprocess_image(img: &DynamicImage) -> DynamicImage {
        let resized = img.resize_exact(
            TARGET_SIZE.0,
            TARGET_SIZE.1,
            image::imageops::FilterType::Lanczos3
        );
        resized
    }

    /// Convertit une image en niveau de gris en vecteur normalisé
    pub fn image_to_vector(img: &DynamicImage) -> DVector<f64> {
        let rgb_img = img.to_rgb8();

        let resized = image::imageops::resize(
            &rgb_img,
            TARGET_SIZE.0,
            TARGET_SIZE.1,
            image::imageops::FilterType::Lanczos3
        );

        let mut flat_vec: Vec<f64> = Vec::with_capacity((TARGET_SIZE.0 * TARGET_SIZE.1 * 3) as usize);

        for pixel in resized.pixels() {
            // Normaliser chaque canal RGB
            flat_vec.push(pixel[0] as f64 / 255.0);
            flat_vec.push(pixel[1] as f64 / 255.0);
            flat_vec.push(pixel[2] as f64 / 255.0);
        }

        DVector::from_vec(flat_vec)
    }

    /// Charge une image depuis un chemin et la prétraite
    pub fn load_and_preprocess<P: AsRef<Path>>(path: P) -> Result<DVector<f64>, Box<dyn Error>> {
        let img = image::open(path)?;
        let processed = Self::preprocess_image(&img);
        Ok(Self::image_to_vector(&processed))
    }

    /// Crée un dataset à partir d'un dossier contenant des sous-dossiers pour chaque catégorie
    pub fn create_dataset_from_directory<P: AsRef<Path>>(
        root_dir: P,
        output_path: P,
    ) -> Result<ImageDataset, Box<dyn Error>> {
        let root = root_dir.as_ref();

        let categories: Vec<String> = fs::read_dir(root)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_dir() {
                    path.file_name()?.to_str().map(String::from)
                } else {
                    None
                }
            })
            .collect();

        println!("Found categories: {:?}", categories);

        let category_mapping: HashMap<String, usize> = categories
            .iter()
            .enumerate()
            .map(|(i, cat)| (cat.clone(), i))
            .collect();

        let reverse_mapping: HashMap<usize, String> = category_mapping
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        let mut data = Vec::new();
        let mut labels = Vec::new();

        for (cat_name, cat_index) in &category_mapping {
            let cat_path = root.join(cat_name);

            let image_pattern = format!("{}/*.{{jpg,jpeg,png}}", cat_path.display());
            let image_paths: Vec<PathBuf> =
                glob::glob(&image_pattern)?.filter_map(Result::ok).collect();

            println!(
                "Found {} images in category {}",
                image_paths.len(),
                cat_name
            );

            for img_path in image_paths {
                match Self::load_and_preprocess(&img_path) {
                    Ok(img_vec) => {
                        data.push(img_vec.as_slice().to_vec());
                        labels.push(*cat_index);
                    }
                    Err(e) => {
                        println!("Error processing {}: {}", img_path.display(), e);
                    }
                }
            }
        }

        let dataset = ImageDataset {
            data,
            labels,
            category_mapping,
            reverse_mapping,
        };

        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &dataset)?;

        println!("Dataset saved with {} images", dataset.data.len());

        Ok(dataset)
    }

    /// Charge un dataset depuis un fichier
    pub fn load_dataset<P: AsRef<Path>>(path: P) -> Result<ImageDataset, Box<dyn Error>> {
        let file = File::open(path)?;
        let dataset: ImageDataset = serde_json::from_reader(file)?;
        Ok(dataset)
    }

    /// Divise un dataset en ensembles d'entraînement et de test
    pub fn split_dataset(dataset: &ImageDataset, train_ratio: f64) -> (Vec<usize>, Vec<usize>) {
        let n_samples = dataset.data.len();
        let n_train = (n_samples as f64 * train_ratio) as usize;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let train_indices = indices[..n_train].to_vec();
        let test_indices = indices[n_train..].to_vec();

        (train_indices, test_indices)
    }

    /// Convertit un dataset en matrices pour l'entraînement
    pub fn dataset_to_matrices(
        dataset: &ImageDataset,
        indices: &[usize],
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let n_samples = indices.len();
        if n_samples == 0 {
            return (DMatrix::zeros(0, 0), DMatrix::zeros(0, 0));
        }

        let n_features = dataset.data[0].len();

        let mut x_matrix = DMatrix::zeros(n_samples, n_features);
        let mut y_matrix = DMatrix::zeros(n_samples, 1);

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..n_features {
                x_matrix[(i, j)] = dataset.data[idx][j];
            }
            y_matrix[(i, 0)] = dataset.labels[idx] as f64;
        }

        (x_matrix, y_matrix)
    }
}
