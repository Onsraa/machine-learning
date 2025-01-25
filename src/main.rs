mod graph;
mod plugins;
mod data;

use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use crate::plugins::models::ModelsPlugin;
use crate::plugins::plots::PlotsPlugins;
use crate::plugins::setup::SetupPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SetupPlugin)
        .add_plugins(EguiPlugin)
        .add_plugins(PlotsPlugins)
        .add_plugins(ModelsPlugin)
        .run();
}