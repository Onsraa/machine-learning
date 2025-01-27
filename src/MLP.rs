
use rand::Rng; // For random number generation

// Perceptron Linear Model
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
                if is_classification || l != self.l {
                    self.x[l][j] = total.tanh();
                }
            }
        }
    }

    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.x[self.l][1..].to_vec()
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
                    self.deltas[self.l][j] *= 1.0 - self.x[self.l][j].tanh().powi(2);
                }
            }
            loss /= self.d[self.l] as f64;
            losses.push(loss);

            // Backpropagation
            for l in (2..=self.l).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.w[l][i][j] * self.deltas[l][j];
                    }
                    self.deltas[l - 1][i] = (1.0 - self.x[l - 1][i].tanh().powi(2)) * total;
                }
            }

            // Update weights
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


let model = MyMLP::new(&[2, 2, 1]);

    // Print weights (W)
    println!("Weights (W): {:?}", model.w);

    // Print activations (X)
    println!("Activations (X): {:?}", model.x);

    // Print deltas
    println!("Deltas: {:?}", model.deltas);


let train_losses = model.train(
    &all_dataset_inputs,     // Reference to the dataset inputs
    &all_dataset_expected_outputs, // Reference to the expected outputs
    0.0005,                  // Learning rate
    1_000_000,               // Number of iterations (underscores for readability)
    true,                    // Is classification
);
    