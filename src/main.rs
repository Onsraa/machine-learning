use std::fs;
use csv::Writer;
use MLP::MyMLP;
use std::path::Path;
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::collections::HashMap;

mod MLP;
mod conver_img;


fn main() {
    println!("LOAD IMAGES TO DATASET ");
    let dataset_path = "src/data/dataset"; // Change if needed
    let image_size = (64, 64);  // Resize images to 64x64

    let dataset = conver_img::load_dataset(dataset_path, image_size);

    println!("Loaded {} images", dataset.len());
  
    // Separate inputs and outputs
    let (dataset_inputs, dataset_outputs): (Vec<Vec<f32>>, Vec<Vec<f32>>) = dataset
    .iter()
    .map(|(input, label)| (input.clone(), label.clone())) // Clone input and label vectors
    .unzip();

    println!("After inputs and outputs manually {:?} ", dataset_inputs.len());
    println!("After dataset outputs  {:?} ", dataset_outputs.len());

    // Create an instance of the MLP model (adjust the layers as needed)
    let layer_sizes = vec![28 * 28, 128, 64, 10]; // Example: Input layer for 28x28 images, hidden layers, and output layer (10 classes)
    let mut model = MyMLP::new(&layer_sizes);

    println!("Model Layersizes : {:?}", layer_sizes);
    

    // Training the model (with learning rate 0.01, 1000 iterations)
    let alpha = 0.01;
    let iterations = 1000;
    let is_classification = true;  // Assuming this is a classification task
    let losses = model.train(&dataset_inputs, &dataset_outputs, alpha, iterations, is_classification);

    println!("losses : {:?}", losses);

    // Print first image's features and label for verification
    // if let Some((features, label)) = dataset.get(0) {
    //     println!("First image label: {:?}", label);
    //     println!("Feature vector length: {}", features.len());  // Should be 64x64x3 = 12288
    // }

    //save_images_to_csv("src/data/dataset.csv", &dataset);
}
