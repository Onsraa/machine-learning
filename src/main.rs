mod data;
mod graph;
mod plugins;

use crate::graph::graph::Graph3DPlugin;
use crate::plugins::models::ModelsPlugin;
use crate::plugins::plots::PlotsPlugins;
use crate::plugins::setup::SetupPlugin;
use bevy::{color::palettes::css::*, prelude::*};
use bevy_egui::EguiPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::Srgba(WHITE_SMOKE)))
        .add_plugins(DefaultPlugins)
        // .add_plugins(SetupPlugin)
        // .add_plugins(EguiPlugin)
        // .add_plugins(PlotsPlugins)
        // .add_plugins(ModelsPlugin)
        .add_plugins(Graph3DPlugin)
        .run();
}

//
// use rand::Rng;
//
// // Structure pour notre modèle linéaire
// struct LinearModel {
//     weights: Vec<f64>,
//     bias: f64,
//     learning_rate: f64,
// }
//
// impl LinearModel {
//     // Initialisation du modèle
//     fn new(num_features: usize, learning_rate: f64) -> Self {
//         let mut rng = rand::thread_rng();
//         LinearModel {
//             weights: (0..num_features)
//                 .map(|_| rng.gen_range(-1.0..1.0))
//                 .collect(),
//             bias: rng.gen_range(-1.0..1.0),
//             learning_rate,
//         }
//     }
//
//     // Prédiction pour un exemple
//     fn predict(&self, features: &[f64]) -> f64 {
//         let sum: f64 = features
//             .iter()
//             .zip(self.weights.iter())
//             .map(|(x, w)| x * w)
//             .sum();
//         sum + self.bias
//     }
//
//     // Entraînement sur un batch de données
//     fn train(&mut self, features: &[Vec<f64>], targets: &[f64], epochs: usize) {
//         for _ in 0..epochs {
//             for (x, y) in features.iter().zip(targets.iter()) {
//                 // Faire une prédiction
//                 let prediction = self.predict(x);
//
//                 // Calculer l'erreur
//                 let error = prediction - y;
//
//                 // Mettre à jour les poids
//                 for (w, x_i) in self.weights.iter_mut().zip(x.iter()) {
//                     *w -= self.learning_rate * error * x_i;
//                 }
//
//                 // Mettre à jour le biais
//                 self.bias -= self.learning_rate * error;
//             }
//         }
//     }
// }
//
// // Fonction principale avec test
// fn main() {
//     // Création de données d'exemple pour deux catégories
//     // Catégorie 1 : Points proches de (0, 0)
//     // Catégorie 2 : Points proches de (1, 1)
//     let features = vec![
//         // Catégorie 1
//         vec![0.1, 0.2],
//         vec![0.2, 0.1],
//         vec![0.0, 0.3],
//         // Catégorie 2
//         vec![0.9, 1.1],
//         vec![1.0, 0.9],
//         vec![1.1, 1.0],
//     ];
//
//     let targets = vec![
//         0.0, // Catégorie 1
//         0.0,
//         0.0,
//         1.0, // Catégorie 2
//         1.0,
//         1.0,
//     ];
//
//     // Créer et entraîner le modèle
//     let mut model = LinearModel::new(2, 0.1);
//
//     println!("Avant l'entraînement:");
//     for (i, x) in features.iter().enumerate() {
//         let pred = model.predict(x);
//         println!("Exemple {}: prédit = {:.3}, réel = {}", i, pred, targets[i]);
//     }
//
//     // Entraînement
//     model.train(&features, &targets, 100_000);
//
//     println!("\nAprès l'entraînement:");
//     for (i, x) in features.iter().enumerate() {
//         let pred = model.predict(x);
//         println!("Exemple {}: prédit = {:.3}, réel = {}", i, pred, targets[i]);
//     }
//
//     // Test sur de nouvelles données
//     let test_points = vec![
//         vec![0.15, 0.15], // Devrait être proche de 0.0
//         vec![0.95, 0.95], // Devrait être proche de 1.0
//     ];
//
//     println!("\nPrédictions sur de nouvelles données:");
//     for (i, x) in test_points.iter().enumerate() {
//         let pred = model.predict(x);
//         println!("Point de test {}: prédit = {:.3}", i, pred);
//     }
// }
