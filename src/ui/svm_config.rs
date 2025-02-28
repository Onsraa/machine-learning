use crate::algorithms::model_selector::ModelAlgorithm;
use crate::algorithms::svm::{SVM, KernelType};
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Slider, DragValue, ComboBox};

pub fn svm_config_ui(mut contexts: EguiContexts, mut training_state: ResMut<TrainingState>) {
    // Vérifier si un modèle SVM est sélectionné
    let is_svm_model = match &training_state.selected_model {
        Some(ModelAlgorithm::SVM(_, _)) => true,
        _ => false,
    };

    if !is_svm_model {
        return;
    }

    if let Some(ModelAlgorithm::SVM(ref mut svm_model, _)) = &mut training_state.selected_model {
        egui::Window::new("SVM Configuration").show(contexts.ctx_mut(), |ui| {
            ui.heading("SVM Hyperparameters");
            ui.separator();

            ui.label("Kernel Type:");
            let kernel_names = ["Linear", "Polynomial", "RBF"];
            let current_kernel = match svm_model.kernel_type {
                KernelType::Linear => 0,
                KernelType::Polynomial => 1,
                KernelType::RBF => 2,
            };
            let mut selected_kernel = current_kernel;

            ComboBox::from_label("")
                .selected_text(kernel_names[current_kernel])
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected_kernel, 0, "Linear");
                    ui.selectable_value(&mut selected_kernel, 1, "Polynomial");
                    ui.selectable_value(&mut selected_kernel, 2, "RBF");
                });

            if selected_kernel != current_kernel {
                match selected_kernel {
                    0 => svm_model.kernel_type = KernelType::Linear,
                    1 => svm_model.kernel_type = KernelType::Polynomial,
                    2 => svm_model.kernel_type = KernelType::RBF,
                    _ => {}
                }
                println!("Changed kernel type to {:?}", svm_model.kernel_type);
            }
            ui.separator();

            match svm_model.kernel_type {
                KernelType::Polynomial => {
                    ui.horizontal(|ui| {
                        ui.label("Polynomial Degree:");
                        let mut degree = svm_model.polynomial_degree as i32;
                        if ui.add(DragValue::new(&mut degree).range(1..=10)).changed() {
                            svm_model.polynomial_degree = degree as usize;
                            println!("Changed polynomial degree to {}", svm_model.polynomial_degree);
                        }
                    });
                },
                KernelType::RBF => {
                    ui.horizontal(|ui| {
                        ui.label("Gamma (RBF width):");
                        if ui.add(Slider::new(&mut svm_model.gamma, 0.01..=10.0).logarithmic(true)).changed() {
                            println!("Changed gamma to {}", svm_model.gamma);
                        }
                    });
                    ui.label("• Low gamma (0.01-0.1): Wider influence, smoother function");
                    ui.label("• Medium gamma (0.1-1.0): Balanced local/global influence");
                    ui.label("• High gamma (1.0-10.0): Very local influence");
                },
                _ => {}
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("C (regularization):");
                if ui.add(Slider::new(&mut svm_model.c, 0.1..=100.0).logarithmic(true)).changed() {
                    println!("Changed C to {}", svm_model.c);
                }
            });
            ui.label("• Low C (0.1-1.0): More regularization, smoother boundary");
            ui.label("• High C (10-100): Less regularization, fits training data closer");
            ui.separator();

            ui.collapsing("Advanced Parameters", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Tolerance:");
                    if ui.add(Slider::new(&mut svm_model.tolerance, 1e-5..=1e-1).logarithmic(true)).changed() {
                        println!("Changed tolerance to {}", svm_model.tolerance);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Max Iterations:");
                    let mut max_iter = svm_model.max_iterations as i32;
                    if ui.add(DragValue::new(&mut max_iter).range(10..=10000)).changed() {
                        svm_model.max_iterations = max_iter as usize;
                        println!("Changed max iterations to {}", svm_model.max_iterations);
                    }
                });
            });

            ui.separator();
            ui.heading("Model Info");

            if let Some(sv) = &svm_model.support_vectors {
                ui.label(format!("Support Vectors: {}", sv.nrows()));
                ui.label(format!("Bias: {:.6}", svm_model.bias));
            } else {
                ui.colored_label(egui::Color32::YELLOW, "Model not trained yet");
            }
        });
    }
}