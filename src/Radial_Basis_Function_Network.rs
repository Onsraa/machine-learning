//RBF Network
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::collections::HashMap;

// Radial Basis Function (Gaussian)
fn gaussian_rbf(x: &DVector<f64>, center: &DVector<f64>, width: f64) -> f64 {
    let diff = x - center;
    (-diff.norm_squared() / (2.0 * width * width)).exp()
}

pub struct RBFNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    centers: Vec<DVector<f64>>,
    widths: Vec<f64>,
    weights: DMatrix<f64>,
}

impl RBFNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            centers: Vec::new(),
            widths: Vec::new(),
            weights: DMatrix::zeros(output_size, hidden_size + 1), // +1 for bias
        }
    }

    // K-Means clustering for center initialization (simplified)
    pub fn initialize_centers(&mut self, data: &[DVector<f64>]) {
        let num_samples = data.len();
        let mut rng = rand::thread_rng();

        // Randomly choose initial centers
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut rng);
        let initial_center_indices = &indices[..self.hidden_size];

        self.centers = initial_center_indices.iter().map(|&i| data[i].clone()).collect();

        // Simple width calculation (average distance to nearest neighbor)
        self.widths = self.centers.iter().map(|center| {
            let mut distances: Vec<f64> = data.iter().map(|x| (x - center).norm()).collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if distances.len() > 1 {
                distances[1] // Nearest neighbor (excluding self)
            } else {
                1.0 // Default width if only one data point
            }
        }).collect();
    }


    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut hidden_activations = DVector::zeros(self.hidden_size + 1);
        hidden_activations[0] = 1.0; // Bias term

        for i in 0..self.hidden_size {
            hidden_activations[i + 1] = gaussian_rbf(input, &self.centers[i], self.widths[i]);
        }
        &self.weights * hidden_activations
    }

    pub fn train(&mut self, data: &[DVector<f64>], labels: &[usize], learning_rate: f64, epochs: usize) {

        let num_samples = data.len();

        for _ in 0..epochs {
            for i in 0..num_samples {
                let input = &data[i];
                let target_label = labels[i];

                let output = self.forward(input);

                // One-hot encoding for target
                let mut target = DVector::zeros(self.output_size);
                target[target_label] = 1.0;

                let error = &target - output;

                // Update weights (using a simple gradient descent)
                let mut hidden_activations = DVector::zeros(self.hidden_size + 1);
                hidden_activations[0] = 1.0;

                for j in 0..self.hidden_size {
                    hidden_activations[j + 1] = gaussian_rbf(input, &self.centers[j], self.widths[j]);
                }

                self.weights += learning_rate * &error * hidden_activations.transpose();
            }
        }
    }

    pub fn predict(&self, input: &DVector<f64>) -> usize {
        let output = self.forward(input);
        output.argmax().0 // Returns the index of the maximum value
    }
}
