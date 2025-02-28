use crate::algorithms::model_selector::ModelAlgorithm;
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{DragValue, Slider};
use rand::Rng;

pub fn rbf_config_ui(mut contexts: EguiContexts, mut training_state: ResMut<TrainingState>) {
    let is_rbf_model = match &training_state.selected_model {
        Some(ModelAlgorithm::RBF(_, _)) => true,
        _ => false,
    };

    if !is_rbf_model {
        return;
    }

    let is_classification = match &training_state.selected_model {
        Some(model) => model.is_classification(),
        None => false,
    };

    if let Some(ModelAlgorithm::RBF(ref mut rbf_model, _)) = &mut training_state.selected_model {
        egui::Window::new("RBF Configuration").show(contexts.ctx_mut(), |ui| {
            ui.heading("RBF Hyperparameters");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Model type:");
                ui.colored_label(
                    if is_classification {
                        egui::Color32::GOLD
                    } else {
                        egui::Color32::LIGHT_BLUE
                    },
                    if is_classification {
                        "Classification"
                    } else {
                        "Regression"
                    },
                );
            });
            ui.separator();

            ui.label("Number of centers:");
            let current_centers = rbf_model.centers.nrows() as u32;
            let mut new_centers = current_centers;
            if ui
                .add(DragValue::new(&mut new_centers).speed(1).range(1..=100))
                .changed()
            {
                println!(
                    "RBF: Changed centers from {} to {}",
                    current_centers, new_centers
                );
                let input_dim = rbf_model.centers.ncols();
                let mut rng = rand::thread_rng();
                rbf_model.centers =
                    nalgebra::DMatrix::from_fn(new_centers as usize, input_dim, |_, _| {
                        rng.gen_range(-1.0..1.0)
                    });
                rbf_model.weights = nalgebra::DVector::from_fn(new_centers as usize, |_, _| {
                    rng.gen_range(-0.1..0.1)
                });
                println!("RBF: Updated centers and weights.");
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Gamma:");
                if ui
                    .add(Slider::new(&mut rbf_model.gamma, 0.01..=10.0).logarithmic(true))
                    .changed()
                {
                    println!("RBF: Changed gamma to {}", rbf_model.gamma);
                }
            });

            ui.label("Guidance:");
            ui.label("• Low gamma (0.01-0.1): Wider influence, smoother function");
            ui.label("• Medium gamma (0.1-1.0): Balanced local/global influence");
            ui.label("• High gamma (1.0-10.0): Very local influence, can model sharp changes");
            ui.separator();

            let mut is_class = rbf_model.is_classification;
            if ui.checkbox(&mut is_class, "Classification model").changed() {
                println!(
                    "RBF: Changed is_classification from {} to {}",
                    rbf_model.is_classification, is_class
                );
                rbf_model.is_classification = is_class;
                ui.label("⚠️ Classification change will apply after closing this window");
            }
            ui.separator();

            let mut new_update = rbf_model.update_centers;
            if ui
                .checkbox(
                    &mut new_update,
                    "Use K-means to select centers during training",
                )
                .changed()
            {
                println!(
                    "RBF: Changed update_centers from {} to {}",
                    rbf_model.update_centers, new_update
                );
                rbf_model.update_centers = new_update;
            }

            if rbf_model.update_centers {
                ui.horizontal(|ui| {
                    ui.label("K-means max iterations:");
                    let mut new_max_iters = rbf_model.max_kmeans_iters as u32;
                    if ui
                        .add(
                            DragValue::new(&mut new_max_iters)
                                .speed(10)
                                .range(10..=1000),
                        )
                        .changed()
                    {
                        println!(
                            "RBF: Changed max_kmeans_iters from {} to {}",
                            rbf_model.max_kmeans_iters, new_max_iters
                        );
                        rbf_model.max_kmeans_iters = new_max_iters as usize;
                    }
                });
            }
        });

        if rbf_model.is_classification != is_classification {
            let rbf_clone = rbf_model.clone();
            training_state.selected_model = Some(ModelAlgorithm::new_rbf(rbf_clone));
            println!("RBF: Updated model TaskType to match classification status");
        }
    }
}
