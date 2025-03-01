mod algorithms;
mod components;
mod data;
mod params;
mod plugins;
mod resources;
mod systems;
mod ui;

use crate::plugins::models::ModelsPlugin;
use crate::plugins::setup::SetupPlugin;
use crate::ui::plugin::UiPlugin;
use bevy::{color::palettes::css::*, prelude::*};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::Srgba(WHITE_SMOKE)))
        .add_plugins(DefaultPlugins)
        .add_plugins(SetupPlugin)
        .add_plugins(ModelsPlugin)
        .add_plugins(UiPlugin)
        .run();
}
