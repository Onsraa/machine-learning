mod components;
mod data;
mod plugins;
mod systems;
mod ui;
mod algorithms;

use crate::plugins::models::ModelsPlugin;
use crate::plugins::setup::SetupPlugin;
use crate::ui::plugin::UiPlugin;
use bevy::{color::palettes::css::*, prelude::*};
use crate::plugins::linear_model::LinearModelPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::Srgba(WHITE_SMOKE)))
        .add_plugins(DefaultPlugins)
        .add_plugins(SetupPlugin)
        .add_plugins(ModelsPlugin)
        .add_plugins(UiPlugin)
        .add_plugins(LinearModelPlugin)
        .run();
}
