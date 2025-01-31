use crate::data::models::{DataModel, ModelState, Points};
use bevy::prelude::*;
use bevy_egui::EguiContexts;

pub fn update_ui(
    mut commands: Commands,
    mut points: ResMut<Points>,
    mut data_model: ResMut<DataModel>,
    mut contexts: EguiContexts,
    current_state: Res<State<ModelState>>,
    mut next_state: ResMut<NextState<ModelState>>,
) {
    egui::Window::new("Control panel").show(contexts.ctx_mut(), |ui| {
        ui.collapsing("Models", |ui| {
            if ui.button("Linear Simple").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::LinearSimple => {}
                        _ => {
                            commands.insert_resource(DataModel::LinearSimple);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Linear Multiple").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::LinearMultiple => {}
                        _ => {
                            commands.insert_resource(DataModel::LinearMultiple);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("XOR").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::XOR => {}
                        _ => {
                            commands.insert_resource(DataModel::XOR);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Cross").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::Cross => {}
                        _ => {
                            commands.insert_resource(DataModel::Cross);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Multi Linear 3 Classes").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::MultiLinear3Classes => {}
                        _ => {
                            commands.insert_resource(DataModel::MultiLinear3Classes);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Multi Cross").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::MultiCross => {}
                        _ => {
                            commands.insert_resource(DataModel::MultiCross);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Linear Simple 2d").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::LinearSimple2d => {}
                        _ => {
                            commands.insert_resource(DataModel::LinearSimple2d);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Linear Simple 3d").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::LinearSimple3d => {}
                        _ => {
                            commands.insert_resource(DataModel::LinearSimple3d);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Linear Tricky 3d").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::LinearTricky3d => {}
                        _ => {
                            commands.insert_resource(DataModel::LinearTricky3d);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Non Linear Simple 2d").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::NonLinearSimple2d => {}
                        _ => {
                            commands.insert_resource(DataModel::NonLinearSimple2d);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
            if ui.button("Non Linear Simple 3d").clicked() {
                match current_state.get() {
                    ModelState::Ready => match *data_model {
                        DataModel::NonLinearSimple3d => {}
                        _ => {
                            commands.insert_resource(DataModel::NonLinearSimple3d);
                            next_state.set(ModelState::Updating)
                        }
                    },
                    _ => {}
                }
            }
        });
    });
}
