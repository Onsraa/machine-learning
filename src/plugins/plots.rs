use bevy::app::*;
use crate::graph::egui::*;

pub struct PlotsPlugins;

impl Plugin for PlotsPlugins {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_plot);
    }
}