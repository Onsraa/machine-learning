use crate::algorithms::learning_model::LearningModel;
use nalgebra::{DMatrix, DVector};
use std::result::Result;
use std::f64;
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum KernelType {
    Linear,
    Polynomial,
    RBF,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SVM {
    pub input_dim: usize,
    pub alphas: DVector<f64>,
    pub bias: f64,
    pub support_vectors: Option<DMatrix<f64>>,
    pub support_vector_labels: Option<DVector<f64>>,
    pub kernel_type: KernelType,
    pub polynomial_degree: usize,
    pub gamma: f64,
    pub c: f64,
    pub is_classification: bool,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl SVM {
    pub fn new(
        input_dim: usize,
        kernel_type: KernelType,
        polynomial_degree: usize,
        gamma: f64,
        c: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Self {
        if c <= 0.0 {
            panic!("C must be positive");
        }
        if gamma <= 0.0 && kernel_type == KernelType::RBF {
            panic!("Gamma must be positive for RBF kernel");
        }
        if polynomial_degree == 0 && kernel_type == KernelType::Polynomial {
            panic!("Polynomial degree must be positive");
        }

        Self {
            input_dim,
            alphas: DVector::zeros(0),
            bias: 0.0,
            support_vectors: None,
            support_vector_labels: None,
            kernel_type,
            polynomial_degree,
            gamma,
            c,
            is_classification: true,
            tolerance,
            max_iterations,
        }
    }

    fn kernel(&self, x1: &DVector<f64>, x2: &DVector<f64>) -> f64 {
        match self.kernel_type {
            KernelType::Linear => x1.dot(x2),
            KernelType::Polynomial => {
                let dot_product = x1.dot(x2);
                (1.0 + dot_product).powf(self.polynomial_degree as f64)
            }
            KernelType::RBF => {
                let diff = x1 - x2;
                let squared_norm = diff.dot(&diff);
                (-self.gamma * squared_norm).exp()
            }
        }
    }

    fn compute_kernel_matrix(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n_samples = x.nrows();
        let mut kernel_matrix = DMatrix::zeros(n_samples, n_samples);

        for i in 0..n_samples {
            let x_i = x.row(i).transpose();
            for j in 0..n_samples {
                let x_j = x.row(j).transpose();
                kernel_matrix[(i, j)] = self.kernel(&x_i, &x_j);
            }
        }

        kernel_matrix
    }

    fn smo(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<(), String> {
        let n_samples = x.nrows();

        // Initialiser les alphas et le biais
        self.alphas = DVector::zeros(n_samples);
        self.bias = 0.0;

        // Calculer la matrice de noyau (Gram matrix)
        let kernel_matrix = self.compute_kernel_matrix(x);

        // Variables pour suivre les changements
        let mut num_changed_alphas = 0;
        let mut examine_all = true;
        let mut iteration = 0;

        while (num_changed_alphas > 0 || examine_all) && iteration < self.max_iterations {
            num_changed_alphas = 0;

            for i in 0..n_samples {
                if !examine_all && self.alphas[i] <= 0.0 && self.alphas[i] >= self.c {
                    continue;
                }

                let error_i = self.predict_value(x, &kernel_matrix, i) - y[i];

                let r_i = error_i * y[i];

                if (r_i < -self.tolerance && self.alphas[i] < self.c) ||
                    (r_i > self.tolerance && self.alphas[i] > 0.0) {

                    let mut j = (i + 1) % n_samples;
                    if i == j {
                        j = (j + 1) % n_samples;
                    }

                    let error_j = self.predict_value(x, &kernel_matrix, j) - y[j];

                    let alpha_i_old = self.alphas[i];
                    let alpha_j_old = self.alphas[j];

                    let (l, h) = if y[i] == y[j] {
                        (f64::max(0.0, self.alphas[j] + self.alphas[i] - self.c),
                         f64::min(self.c, self.alphas[j] + self.alphas[i]))
                    } else {
                        (f64::max(0.0, self.alphas[j] - self.alphas[i]),
                         f64::min(self.c, self.c + self.alphas[j] - self.alphas[i]))
                    };

                    if f64::abs(l - h) < 1e-4 {
                        continue;
                    }

                    let eta = 2.0 * kernel_matrix[(i, j)] - kernel_matrix[(i, i)] - kernel_matrix[(j, j)];

                    if eta >= 0.0 {
                        continue;
                    }

                    self.alphas[j] = alpha_j_old - (y[j] * (error_i - error_j)) / eta;

                    self.alphas[j] = f64::min(f64::max(self.alphas[j], l), h);

                    if f64::abs(self.alphas[j] - alpha_j_old) < 1e-5 {
                        continue;
                    }

                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alphas[j]);

                    let b1 = self.bias - error_i -
                        y[i] * (self.alphas[i] - alpha_i_old) * kernel_matrix[(i, i)] -
                        y[j] * (self.alphas[j] - alpha_j_old) * kernel_matrix[(i, j)];

                    let b2 = self.bias - error_j -
                        y[i] * (self.alphas[i] - alpha_i_old) * kernel_matrix[(i, j)] -
                        y[j] * (self.alphas[j] - alpha_j_old) * kernel_matrix[(j, j)];

                    if 0.0 < self.alphas[i] && self.alphas[i] < self.c {
                        self.bias = b1;
                    } else if 0.0 < self.alphas[j] && self.alphas[j] < self.c {
                        self.bias = b2;
                    } else {
                        self.bias = (b1 + b2) / 2.0;
                    }

                    num_changed_alphas += 1;
                }
            }

            if examine_all {
                examine_all = false;
            } else if num_changed_alphas == 0 {
                examine_all = true;
            }

            iteration += 1;
        }

        let mut support_vectors = Vec::new();
        let mut support_vector_labels = Vec::new();
        let mut support_vector_alphas = Vec::new();

        for i in 0..n_samples {
            if self.alphas[i] > 1e-6_f64 {
                support_vectors.push(x.row(i).clone_owned());
                support_vector_labels.push(y[i]);
                support_vector_alphas.push(self.alphas[i]);
            }
        }

        if support_vectors.is_empty() {
            return Err("No support vectors found. Try adjusting C or using a different kernel.".to_string());
        }

        let n_sv = support_vectors.len();
        let mut sv_matrix = DMatrix::zeros(n_sv, x.ncols());
        for i in 0..n_sv {
            for j in 0..x.ncols() {
                sv_matrix[(i, j)] = support_vectors[i][j];
            }
        }

        self.support_vectors = Some(sv_matrix);
        self.support_vector_labels = Some(DVector::from_vec(support_vector_labels));
        self.alphas = DVector::from_vec(support_vector_alphas);

        println!("SVM training completed with {} support vectors out of {} samples", n_sv, n_samples);
        println!("Final bias: {}", self.bias);

        Ok(())
    }

    fn predict_value(&self, x: &DMatrix<f64>, kernel_matrix: &DMatrix<f64>, idx: usize) -> f64 {
        let mut sum = 0.0;

        for i in 0..self.alphas.len() {
            if self.alphas[i] > 0.0 {
                sum += self.alphas[i] * x.row(i)[0] * kernel_matrix[(i, idx)];
            }
        }

        sum + self.bias
    }

    pub fn predict_one(&self, x: &DVector<f64>) -> f64 {
        if self.support_vectors.is_none() || self.support_vector_labels.is_none() {
            return 0.0;
        }

        let sv = self.support_vectors.as_ref().unwrap();
        let sv_y = self.support_vector_labels.as_ref().unwrap();

        let mut decision_value = 0.0;

        for i in 0..self.alphas.len() {
            let sv_i = sv.row(i).transpose();
            decision_value += self.alphas[i] * sv_y[i] * self.kernel(&sv_i, x);
        }

        decision_value + self.bias
    }
}

impl LearningModel for SVM {
    fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        _learning_rate: f64,
        _n_epochs: usize,
    ) -> Result<Vec<f64>, String> {
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

        let mut y_proc = DVector::zeros(y.nrows());
        for i in 0..y.nrows() {
            if y[(i, 0)] <= 0.0 {
                y_proc[i] = -1.0;
            } else {
                y_proc[i] = 1.0;
            }
        }

        match self.smo(x, &y_proc) {
            Ok(_) => {
                let loss = if let Some(sv) = &self.support_vectors {
                    sv.nrows() as f64 / x.nrows() as f64
                } else {
                    1.0
                };

                Ok(vec![loss])
            },
            Err(e) => Err(e),
        }
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

        let mut correct = 0;
        let n_samples = x.nrows();

        for i in 0..n_samples {
            let x_i = x.row(i).transpose();
            let prediction = self.predict_one(&x_i);
            let actual = if y[(i, 0)] <= 0.0 { -1.0 } else { 1.0 };

            let predicted_class = if prediction >= 0.0 { 1.0 } else { -1.0 };

            if f64::abs(predicted_class - actual) < 1e-10 {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / n_samples as f64;
        Ok(1.0 - accuracy)
    }

    fn predict(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        if self.support_vectors.is_none() {
            return Err("Model not trained yet".to_string());
        }

        let decision_value = self.predict_one(x);
        let class = if decision_value >= 0.0 { 1.0 } else { 0.0 };

        Ok(DVector::from_element(1, class))
    }
}