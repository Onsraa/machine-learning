use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::Rng;
use crate::algorithms::learning_model::LearningModel;
use std::result::Result;

#[derive(Clone)]
pub struct RBF {
    pub centers: DMatrix<f64>,
    pub weights: DVector<f64>,
    pub bias: f64,
    pub gamma: f64,
    pub is_classification: bool,
    pub update_centers: bool,
    pub max_kmeans_iters: usize,
}

impl RBF {
    /// Creates a new RBF model.
    /// - `input_dim`: input dimension.
    /// - `n_centers`: number of centers (K).
    /// - `gamma`: width parameter. Higher values give narrower bell curves.
    ///   Typical values range from 0.1 to 10, depending on data scale.
    /// - `is_classification`: true for classification, false for regression.
    /// - `update_centers`: if true, elects centers via K-means during fit.
    /// - `max_kmeans_iters`: maximum iterations for K-means algorithm.
    pub fn new(input_dim: usize, n_centers: usize, gamma: f64, is_classification: bool, update_centers: bool, max_kmeans_iters: usize) -> Self {
        if gamma <= 0.0 {
            panic!("Gamma must be positive");
        }

        let mut rng = rand::thread_rng();
        // Initialize centers in range [-1, 1]
        let centers = DMatrix::from_fn(n_centers, input_dim, |_, _| rng.gen_range(-1.0..1.0));

        // Initialize weights with small random values scaled by 1/sqrt(n_centers)
        let scale = 1.0 / (n_centers as f64).sqrt();
        let weights = DVector::from_fn(n_centers, |_, _| rng.gen_range(-scale..scale));

        let bias = rng.gen_range(-0.1..0.1);
        Self {
            centers,
            weights,
            bias,
            gamma,
            is_classification,
            update_centers,
            max_kmeans_iters,
        }
    }

    /// Lloyd's algorithm (K-means) to elect centers from data.
    fn k_means(data: &DMatrix<f64>, k: usize, max_iters: usize) -> DMatrix<f64> {
        let n_samples = data.nrows();
        let dim = data.ncols();
        let effective_k = k.min(n_samples);

        // Initialize centers by randomly selecting k data points
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        let mut centers = DMatrix::from_fn(effective_k, dim, |i, j| data[(indices[i], j)]);

        // Pre-allocate clusters and new_centers matrices
        let mut clusters: Vec<Vec<usize>> = vec![vec![]; effective_k];
        let mut new_centers = DMatrix::zeros(effective_k, dim);

        for iter in 0..max_iters {
            // Clear previous cluster assignments
            for cluster in &mut clusters {
                cluster.clear();
            }

            // Assignment step: find closest center for each point
            for i in 0..n_samples {
                let point = data.row(i);
                let mut best = 0;
                let mut best_dist = f64::INFINITY;

                for j in 0..effective_k {
                    let center = centers.row(j);
                    let mut sum = 0.0;
                    for d in 0..dim {
                        let diff = point[d] - center[d];
                        sum += diff * diff;
                    }
                    if sum < best_dist {
                        best_dist = sum;
                        best = j;
                    }
                }

                clusters[best].push(i);
            }

            // Update step: recalculate centers
            new_centers.fill(0.0);
            let mut converged = true;

            for j in 0..effective_k {
                if clusters[j].is_empty() {
                    // If cluster is empty, keep old center
                    new_centers.row_mut(j).copy_from(&centers.row(j));
                } else {
                    // Calculate new center as mean of assigned points
                    for d in 0..dim {
                        let mut sum = 0.0;
                        for &i in &clusters[j] {
                            sum += data[(i, d)];
                        }
                        new_centers[(j, d)] = sum / (clusters[j].len() as f64);
                    }

                    // Check convergence
                    let mut diff = 0.0;
                    for d in 0..dim {
                        diff += (new_centers[(j, d)] - centers[(j, d)]).abs();
                    }
                    if diff > 1e-6 {
                        converged = false;
                    }
                }
            }

            centers.copy_from(&new_centers);

            // Early stopping if converged
            if converged {
                println!("K-means converged after {} iterations", iter + 1);
                break;
            }
        }

        centers
    }

    /// Calculate RBF activations for a given input.
    /// φᵢ(x) = exp(–γ * ||x – centerᵢ||²)
    pub fn compute_phi(&self, x: &DVector<f64>) -> DVector<f64> {
        let n_centers = self.centers.nrows();
        let mut phi = DVector::zeros(n_centers);

        for i in 0..n_centers {
            let center = self.centers.row(i).transpose();
            let diff = x - center;
            let norm_sq = diff.dot(&diff);
            phi[i] = (-self.gamma * norm_sq).exp();
        }

        phi
    }

    /// Calculate model output for a given input.
    pub fn forward(&self, x: &DVector<f64>) -> f64 {
        let phi = self.compute_phi(x);
        self.weights.dot(&phi) + self.bias
    }
}

impl LearningModel for RBF {
    fn fit(&mut self, x: &DMatrix<f64>, y: &DMatrix<f64>, learning_rate: f64, n_epochs: usize) -> Result<Vec<f64>, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if n_epochs == 0 {
            return Err("Number of epochs must be positive".to_string());
        }
        if x.nrows() == 0 || y.nrows() == 0 {
            return Err("Empty dataset".to_string());
        }
        if x.nrows() != y.nrows() {
            return Err(format!(
                "Number of samples in x ({}) doesn't match y ({})",
                x.nrows(),
                y.nrows()
            ));
        }
        if y.ncols() != 1 {
            return Err(format!("Expected y to have 1 column, got {}", y.ncols()));
        }

        // Update centers via K-means, ensuring k doesn't exceed number of samples
        if self.update_centers {
            let effective_k = self.centers.nrows().min(x.nrows());
            self.centers = Self::k_means(x, effective_k, self.max_kmeans_iters);

            let mut rng = rand::thread_rng();
            // Reinitialize weights and bias based on new number of centers
            let scale = 1.0 / (effective_k as f64).sqrt();
            self.weights = DVector::from_fn(effective_k, |_, _| rng.gen_range(-scale..scale));
            self.bias = rng.gen_range(-0.1..0.1);
        }

        let n_samples = x.nrows();
        let mut losses = Vec::with_capacity(n_epochs);

        for _ in 0..n_epochs {
            let mut total_loss = 0.0;

            for i in 0..n_samples {
                let x_i = DVector::from_iterator(x.ncols(), x.row(i).iter().cloned());
                let target = y[(i, 0)];
                let output = self.forward(&x_i);
                let error = output - target;

                total_loss += error * error;

                // Update weights and bias
                let phi = self.compute_phi(&x_i);
                self.weights -= learning_rate * error * phi;
                self.bias -= learning_rate * error;
            }

            losses.push(total_loss / n_samples as f64);
        }

        Ok(losses)
    }

    fn evaluate(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64, String> {
        if x.nrows() == 0 || y.nrows() == 0 {
            return Err("Empty dataset".to_string());
        }
        if x.nrows() != y.nrows() {
            return Err(format!(
                "Number of samples in x ({}) doesn't match y ({})",
                x.nrows(),
                y.nrows()
            ));
        }
        if y.ncols() != 1 {
            return Err(format!("Expected y to have 1 column, got {}", y.ncols()));
        }

        let n_samples = x.nrows();
        let mut total_loss = 0.0;

        for i in 0..n_samples {
            let x_i = DVector::from_iterator(x.ncols(), x.row(i).iter().cloned());
            let target = y[(i, 0)];
            let output = self.forward(&x_i);
            let error = output - target;
            total_loss += error * error;
        }

        Ok(total_loss / n_samples as f64)
    }

    fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        let output = self.forward(x);

        if self.is_classification {
            let label = if output >= 0.0 { 1.0 } else { 0.0 };
            Ok(DVector::from_element(1, label))
        } else {
            Ok(DVector::from_element(1, output))
        }
    }
}