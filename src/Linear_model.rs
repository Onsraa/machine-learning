use rand::Rng; // For random number generation

// Points array
let points: Vec<Vec<f64>> = vec![
    vec![1.0, 1.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
];

// Labels array
let labels: Vec<Vec<f64>> = vec![
    vec![1.0],
    vec![-1.0],
    vec![-1.0],
];

fn predict(W: &[f64], X: &[f64]) -> f64 {
    let signal: f64 = W[1..].iter().zip(X.iter()).map(|(w, x)| w * x).sum::<f64>() + W[0];
    if signal >= 0.0 { 1.0 } else { -1.0 }
}


fn train(
    w: &mut Vec<f64>,        // Weight vector, passed as mutable reference
    x: &[Vec<f64>],          // Input dataset
    y: &[f64],               // Output labels
    alpha: f64,              // Learning rate
    iteration_count: usize,  // Number of iterations
    points: &[Vec<f64>],     // Points for display_separation
    colors: &[String],       // Colors for display_separation
) {
    let mut rng = rand::thread_rng();

    for _ in 0..iteration_count {
        // Pick a random index
        let k = rng.gen_range(0..x.len());
        let xk = &x[k];
        let yk = y[k];

        // Compute prediction
        let gxk = predict(w, xk);

        // Update weights
        w[0] += alpha * (yk - gxk) * 1.0; // Update bias term
        for i in 0..xk.len() {
            w[i + 1] += alpha * (yk - gxk) * xk[i]; // Update remaining weights
        }

        // Display separation
        display_separation(w, points, colors);
    }
}
