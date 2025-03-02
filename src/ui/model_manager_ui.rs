use crate::resources::model_managers::{ModelManager};
use crate::resources::training::TrainingState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Color32, RichText, ScrollArea};

pub fn model_manager_ui(
    mut contexts: EguiContexts,
    mut model_manager: ResMut<ModelManager>,
    mut training_state: ResMut<TrainingState>,
    time: Res<Time>,
) {
    model_manager.update_status(time.delta_secs());
    let mut selected_index = None;
    let mut load_model = false;
    let mut refresh_list = false;
    let mut show_save_dialog = false;
    let mut request_delete_for_index = None;

    egui::Window::new("Model Manager")
        .default_width(600.0)
        .show(contexts.ctx_mut(), |ui| {
            ui.horizontal(|ui| {
                if ui.button("💾 Save Current Model").clicked() {
                    if training_state.selected_model.is_some() {
                        show_save_dialog = true;
                    } else {
                        model_manager.set_status("No model to save!".to_string(), 3.0);
                    }
                }

                if ui.button("📂 Load Selected Model").clicked() {
                    if model_manager.selected_model_index.is_some() {
                        load_model = true;
                    } else {
                        model_manager.set_status("No model selected!".to_string(), 3.0);
                    }
                }

                if ui.button("🔄 Refresh List").clicked() {
                    refresh_list = true;
                }
            });

            ui.separator();

            if let Some((ref message, _)) = model_manager.status_message {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Status: ").color(Color32::LIGHT_BLUE));
                    ui.label(message);
                });
                ui.separator();
            }

            ui.heading("📋 Saved Models");

            if model_manager.model_infos.is_empty() {
                ui.label("No saved models found");
            } else {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Name").strong());
                    ui.add_space(200.0);
                    ui.label(RichText::new("Type").strong());
                    ui.add_space(100.0);
                    ui.label(RichText::new("Action").strong());
                });

                ui.separator();

                ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                    let mut selected_index_temp = None;
                    let mut delete_index = None;

                    for (i, info) in model_manager.model_infos.iter().enumerate() {
                        if info.category == "cas_de_tests" {
                            let is_selected = model_manager.selected_model_index == Some(i);

                            ui.horizontal(|ui| {
                                let name_label = if is_selected {
                                    RichText::new(&info.name)
                                        .strong()
                                        .color(Color32::LIGHT_BLUE)
                                } else {
                                    RichText::new(&info.name)
                                };

                                if ui.selectable_label(is_selected, name_label).clicked() {
                                    selected_index_temp = Some(i);
                                }

                                ui.add_space(50.0);
                                ui.label(format!("{} ({})", info.model_type, info.task_type));

                                ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                                    if ui.button("Delete").clicked() {
                                        delete_index = Some(i);
                                    }
                                });
                            });

                            if i < model_manager.model_infos.len() - 1 {
                                ui.separator();
                            }
                        }
                    }

                    if let Some(index) = selected_index_temp {
                        selected_index = Some(index);
                    }

                    if let Some(index) = delete_index {
                        request_delete_for_index = Some(index);
                    }
                });
            }
        });

    if let Some(index) = request_delete_for_index {
        model_manager.request_delete_confirmation(index);
    }

    if model_manager.confirm_delete_dialog_open {
        if let Some(index) = model_manager.model_to_delete {
            if index < model_manager.model_infos.len() {
                let model_name = model_manager.model_infos[index].name.clone();

                egui::Window::new("Confirm Deletion")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(contexts.ctx_mut(), |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(format!(
                                "Are you sure you want to delete \"{}\"?",
                                model_name
                            ));
                            ui.label("This action cannot be undone.");

                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui.button("Cancel").clicked() {
                                    model_manager.cancel_delete();
                                }

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::RIGHT),
                                    |ui| {
                                        if ui
                                            .button(RichText::new("Delete").color(Color32::RED))
                                            .clicked()
                                        {
                                            if let Err(e) = model_manager.confirm_delete() {
                                                model_manager
                                                    .set_status(format!("Error: {}", e), 3.0);
                                                println!("Erreur lors de la suppression: {}", e);
                                            } else {
                                                println!("Modèle supprimé avec succès!");
                                            }
                                        }
                                    },
                                );
                            });
                        });
                    });
            } else {
                model_manager.cancel_delete();
            }
        }
    }

    if let Some(index) = selected_index {
        model_manager.selected_model_index = Some(index);
    }

    if load_model {
        if let Some(index) = model_manager.selected_model_index {
            if index < model_manager.model_infos.len() {
                let model_name = model_manager.model_infos[index].name.clone();
                match model_manager.load_model(index) {
                    Ok(model) => {
                        training_state.selected_model = Some(model);
                        model_manager.set_status(
                            format!("Model \"{}\" loaded successfully", model_name),
                            3.0,
                        );
                    }
                    Err(e) => {
                        model_manager.set_status(format!("Error loading model: {}", e), 3.0);
                    }
                }
            }
        }
    }

    if refresh_list {
        *model_manager = ModelManager::new();
        model_manager.set_status("Model list refreshed".parse().unwrap(), 2.0);
    }

    if show_save_dialog {
        model_manager.save_dialog_open = true;
        model_manager.dialog_model_name = "My Model".to_string();
    }

    if model_manager.save_dialog_open {
        let mut save_model = false;
        let mut cancel_save = false;
        let mut model_name = model_manager.dialog_model_name.clone();
        let mut description = model_manager.dialog_description.clone();

        egui::Window::new("Save Model")
            .fixed_size([450.0, 250.0])
            .collapsible(false)
            .resizable(false)
            .show(contexts.ctx_mut(), |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Save Model");
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.add(
                        egui::TextEdit::singleline(&mut model_name)
                            .hint_text("Enter model name")
                            .desired_width(300.0),
                    );
                });

                ui.add_space(10.0);

                ui.label("Description (optional):");
                ui.add(
                    egui::TextEdit::multiline(&mut description)
                        .hint_text("Enter description here...")
                        .desired_width(ui.available_width())
                        .desired_rows(5),
                );

                ui.add_space(20.0);

                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                        if ui.button("Save").clicked() {
                            save_model = true;
                        }

                        if ui.button("Cancel").clicked() {
                            cancel_save = true;
                        }
                    });
                });
            });

        model_manager.dialog_model_name = model_name;
        model_manager.dialog_description = description;

        if save_model {
            if let Some(model) = &training_state.selected_model {
                let model_name = model_manager.dialog_model_name.clone();
                let desc_option = if model_manager.dialog_description.is_empty() {
                    None
                } else {
                    Some(model_manager.dialog_description.clone())
                };

                if let Err(e) = model_manager.save_model_with_category(
                    model,
                    &model_name,
                    desc_option,
                    "cas_de_tests"
                ) {
                    model_manager.set_status(format!("Error saving model: {}", e), 3.0);
                }
                model_manager.dialog_description = String::new();
            }
            model_manager.save_dialog_open = false;
        }

        if cancel_save {
            model_manager.save_dialog_open = false;
            model_manager.dialog_description = String::new();
        }
    }
}