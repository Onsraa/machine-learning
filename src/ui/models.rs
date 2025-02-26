use crate::data::{DataModel, ModelState};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Button, Stroke, Color32};

pub fn update_test_case_ui(
    mut contexts: EguiContexts,
    mut data_model: ResMut<DataModel>,
    current_state: Res<State<ModelState>>,
    mut next_state: ResMut<NextState<ModelState>>,
) {
    egui::Window::new("Test Case Selector").show(contexts.ctx_mut(), |ui| {
        ui.label("Choisissez le cas de test:");
        let variants = vec![
            ("Linear Simple", DataModel::LinearSimple),
            ("Linear Multiple", DataModel::LinearMultiple),
            ("XOR", DataModel::XOR),
            ("Cross", DataModel::Cross),
            ("Multi Linear 3 Classes", DataModel::MultiLinear3Classes),
            ("Multi Cross", DataModel::MultiCross),
            ("Linear Simple 2d", DataModel::LinearSimple2d),
            ("Linear Simple 3d", DataModel::LinearSimple3d),
            ("Linear Tricky 3d", DataModel::LinearTricky3d),
            ("Non Linear Simple 2d", DataModel::NonLinearSimple2d),
            ("Non Linear Simple 3d", DataModel::NonLinearSimple3d),
        ];
        for (label, variant) in variants {
            let is_selected = *data_model == variant;
            let btn = Button::new(label)
                .stroke(if is_selected { Stroke::new(2.0, Color32::GOLD) } else { Stroke::new(1.0, Color32::LIGHT_GRAY) });
            if ui.add(btn).clicked() {
                if let ModelState::Ready = *current_state.get() {
                    *data_model = variant;
                    next_state.set(ModelState::Updating);
                }
            }
        }
    });
}
