use rand::Rng; // For random number generation

// Start init Weight Vectors (scalars) W =[W0,W1,W2,...,Wn]
let points: [[f64; 2]; 3] = [
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
];

let labels: [[f64; 1]; 3] = [
    [1.0],
    [-1.0],
    [-1.0],
];

pub struct MyMLP {
    d: Vec<usize>,            // Neurons per layer
    l: usize,                 // Number of layers
    w: Vec<Vec<Vec<f64>>>,    // Weights
    x: Vec<Vec<f64>>,         // Activations
    deltas: Vec<Vec<f64>>,    // Deltas
}

impl MyMLP {
    pub fn new(npl: &[usize]) -> Self {

        let d = npl.to_vec();
        let l = d.len() - 1;

        // Initialize weights
        let mut w = vec![vec![]; d.len()];
        let mut rng = rand::thread_rng();
        for (layer_idx, layer_size) in d.iter().enumerate() {
            if layer_idx == 0 {
                continue; // Input layer has no weights
            }
            w[layer_idx] = vec![vec![0.0; d[layer_idx] + 1]; d[layer_idx - 1] + 1];
            for i in 0..=d[layer_idx - 1] {
                for j in 1..=d[layer_idx] {
                    w[layer_idx][i][j] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        // Initialize activations and deltas
        let mut x = vec![];
        let mut deltas = vec![];
        for &layer_size in &d {
            let mut layer_x = vec![1.0; layer_size + 1];
            let mut layer_deltas = vec![0.0; layer_size + 1];
            for j in 1..=layer_size {
                layer_x[j] = 0.0;
            }
            x.push(layer_x);
            deltas.push(layer_deltas);
        }

        Self { d, l, w, x, deltas }
    }

    //  ReLU function
    fn relu(x: f64) -> f64 {
        x.max(0.0)  // Returns 0 if x < 0, otherwise x
    }

    fn softmax(x: &[f64]) -> Vec<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Find max for stability
        let exp_x: Vec<f64> = x.iter().map(|&val| (val - max_val).exp()).collect();
        let sum_exp_x: f64 = exp_x.iter().sum();
        exp_x.iter().map(|&val| val / sum_exp_x).collect()
    }

    fn propagate(&mut self, inputs: &[f64], is_classification: bool) {
        // Set input layer activations
        for j in 1..=self.d[0] {
            self.x[0][j] = inputs[j - 1];
        }

        // Forward propagation
        for l in 1..=self.l {
            for j in 1..=self.d[l] {
                let mut total = 0.0;
                for i in 0..=self.d[l - 1] {
                    total += self.w[l][i][j] * self.x[l - 1][i];
                }
                self.x[l][j] = total;
                
                if is_classification && l == self.l { 
                    // Apply softmax to the output layer:
                    let output_values: Vec<f64> = self.x[l][1..].to_vec();
                    let softmax_output = Self::softmax(&output_values);
                    for (k, &val) in softmax_output.iter().enumerate() {
                        self.x[l][k + 1] = val;
                    }
                } else if l != self.l { // Hidden layers (still use ReLU)
                    self.x[l][j] = Self::relu(self.x[l][j]);
                }
            }
        }
    }

    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.x[self.l][1..].to_vec()
    }

    // Define ReLU derivative function (for backpropagation)
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn train(
        &mut self,
        dataset_inputs: &[Vec<f64>],
        dataset_outputs: &[Vec<f64>],
        alpha: f64,
        iterations: usize,
        is_classification: bool,
    ) -> Vec<f64> {

        let mut losses = vec![];

        for _ in 0..iterations {
            // Randomly pick a data point
            let k = rand::thread_rng().gen_range(0..dataset_inputs.len());
            let inputs = &dataset_inputs[k];
            let outputs = &dataset_outputs[k];

            // Forward propagate
            self.propagate(inputs, is_classification);

            // Compute loss and output layer deltas
            let mut loss = 0.0;

            for j in 1..=self.d[self.l] {
                let error = self.x[self.l][j] - outputs[j - 1];
                loss += error.powi(2);
                self.deltas[self.l][j] = error;

                if is_classification {
                    // Delta calculation for softmax output
                    self.deltas[self.l][j] = error;
                    //self.deltas[self.l][j] *= Self::relu_derivative(self.x[self.l][j]);
                    //self.deltas[self.l][j] *= 1.0 - self.x[self.l][j].tanh().powi(2);
                }
            }

            // Compute hidden layer deltas
            for l in (2..=self.l).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.w[l][i][j] * self.deltas[l][j];
                    }
                    self.deltas[l - 1][i] = Self::relu_derivative(self.x[l - 1][i]) * total;
                }
            }

            loss /= self.d[self.l] as f64;
            losses.push(loss);
            
            for l in 1..=self.l {
                for i in 0..=self.d[l - 1] {
                    for j in 1..=self.d[l] {
                        self.w[l][i][j] -= alpha * self.x[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
        }
       
        losses
    }
}

// Train model
let mut model = MyMLP::new(&[2, 2, 1]);

let train_losses = model.train(
    &all_dataset_inputs,     // Reference to the dataset inputs
    &all_dataset_expected_outputs, // Reference to the expected outputs
    0.0005,                  // Learning rate
    1_000_000,               // Number of iterations (underscores for readability)
    true,                    // Is classification
);
    