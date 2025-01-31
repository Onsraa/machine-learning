use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use crate::ui::systems::update_ui;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin);
        app.add_systems(Update, update_ui);
    }
}
