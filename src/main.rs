mod data;
mod plugins;
mod components;
mod systems;

use crate::plugins::setup::SetupPlugin;
use crate::plugins::models::ModelsPlugin;
use bevy::{color::palettes::css::*, prelude::*};
use bevy_egui::EguiPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::Srgba(WHITE_SMOKE)))
        .add_plugins(DefaultPlugins)
        .add_plugins(SetupPlugin)
        // .add_plugins(ModelsPlugin)
        // .add_plugins(EguiPlugin)
        .run();
}
