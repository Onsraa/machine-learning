use bevy::prelude::*;
use crate::algorithms::model_linear::convert_points_to_training_data;
use crate::data::models::*;

pub struct ModelsPlugin;

impl Plugin for ModelsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DataModel>();
        app.init_resource::<Points>();
        app.init_resource::<MaxCoordinates>();
        app.init_state::<ModelState>();
        app.add_systems(Update, (update_points, update_max_coordinates, draw_points, convert_points_to_training_data).chain().run_if(in_state(ModelState::Updating)));
    }
}