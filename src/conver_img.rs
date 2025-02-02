use std::fs;
use csv::Writer;
use std::path::Path;
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::collections::HashMap;


pub fn get_label_map() -> HashMap<&'static str, Vec<f32>> {
    let mut labels = HashMap::new();
    labels.insert("RTS", vec![1.0, 0.0, 0.0]);  // One-hot encoding for RTS
    labels.insert("MOBA", vec![0.0, 1.0, 0.0]); // One-hot encoding for MOBA
    labels.insert("FPS", vec![0.0, 0.0, 1.0]); // One-hot encoding for FPS
    labels
}

// convert raw image to compatible format MLP
pub fn image_to_array(image_path: &str, target_size: (u32, u32)) -> Option<Vec<f32>> {
    // Open the image file
    let img = image::open(image_path).ok()?;
    
    // Convert the image to RGB format
    let img = img.resize_exact(target_size.0, target_size.1, image::imageops::FilterType::Triangle);
    let rgb_img = img.to_rgb8();

    // Flatten and normalize pixel values
    let pixels: Vec<f32> = rgb_img
        .pixels()
        .flat_map(|p| vec![p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
        .collect();

    Some(pixels)
}

// Function to scan directories and load dataset
pub fn load_dataset(root_dir: &str, target_size: (u32, u32)) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut dataset = Vec::new();
    let label_map = get_label_map(); // Maps class names to one-hot encoded vectors

    // Iterate over class directories (RTS, MOBA, FPS)
    for entry in fs::read_dir(root_dir).unwrap() {
        if let Ok(entry) = entry {
            let class_path = entry.path();
            if class_path.is_dir() {
                // Extract the class name from the folder name
                if let Some(class_name) = class_path.file_name().and_then(|s| s.to_str()) {
                    if let Some(label_vector) = label_map.get(class_name) {
                        // Iterate over image files in the class folder
                        for img_entry in fs::read_dir(&class_path).unwrap() {
                            if let Ok(img_entry) = img_entry {
                                let img_path = img_entry.path();
                                if img_path.is_file() {
                                    if let Some(feature_vector) = image_to_array(img_path.to_str().unwrap(), target_size) {
                                        dataset.push((feature_vector, label_vector.clone())); // Store the feature vector and label vector
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    dataset
}



/// Function to save dataset to a CSV file
pub fn save_images_to_csv(csv_filename: &str, dataset: &Vec<(Vec<f32>, usize)>) {
    let mut writer = Writer::from_path(csv_filename).expect("Failed to create CSV file");

    for (features, label) in dataset {
        let mut record: Vec<String> = features.iter().map(|&x| x.to_string()).collect();
        record.push(label.to_string()); // Append label at the end
        writer.write_record(&record).expect("Failed to write to CSV");
    }

    writer.flush().expect("Failed to flush CSV writer");
}
