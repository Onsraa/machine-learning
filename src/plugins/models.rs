use bevy::prelude::*;
use crate::data::models::*;

pub struct ModelsPlugin;

impl Plugin for ModelsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DataModel>();
        app.init_resource::<Points>();
        app.add_systems(Startup, set_points);
    }
}