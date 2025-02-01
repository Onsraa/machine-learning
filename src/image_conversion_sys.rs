use std::fs;
use std::path::Path;
use image::{DynamicImage, GenericImageView, ImageFormat};
use csv::Writer;

fn image_to_array(image_path: &str, target_size: (u32, u32)) -> Option<Vec<f32>> {
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

fn save_images_to_csv(folder_path: &str, csv_filename: &str, target_size: (u32, u32)) {
    let paths = fs::read_dir(folder_path).expect("Failed to read directory");
    let mut writer = Writer::from_path(csv_filename).expect("Failed to create CSV file");

    for path in paths {
        if let Ok(entry) = path {
            let file_path = entry.path();
            if file_path.is_file() {
                let ext = file_path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                if ["png", "jpg", "jpeg", "bmp", "gif", "tiff"].contains(&ext.as_str()) {
                    println!("Processing: {:?}", file_path);
                    if let Some(image_data) = image_to_array(file_path.to_str().unwrap(), target_size) {
                        let record: Vec<String> = image_data.iter().map(|&x| x.to_string()).collect();
                        writer.write_record(&record).expect("Failed to write to CSV");
                    }
                }
            }
        }
    }

    writer.flush().expect("Failed to save CSV file");
    println!("Data saved to {}", csv_filename);
}

// example de run de la fonction
fn main() {
    let folder_path = "C:/Users/Tri Uyen/Desktop/ML/machine-learning/src/data/"; // change le chemin ici pour les image
    let csv_filename = "image_data.csv";
    let target_size = (128, 128);
    
    save_images_to_csv(folder_path, csv_filename, target_size);
}