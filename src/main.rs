mod algorithms;
mod graph;
mod parameters;
mod plugins;
mod data;

use plugins::SetupPlugin;

use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use nalgebra::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SetupPlugin)
        .add_plugins(EguiPlugin)
        .run();
}