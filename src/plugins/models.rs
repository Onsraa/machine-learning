use crate::data::*;
use crate::resources::training::TrainingState;
use crate::systems::training::training_system;
use crate::ui::training::training_ui_system;
use bevy::prelude::*;

pub struct ModelsPlugin;

impl Plugin for ModelsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DataModel>();
        app.init_resource::<DatasetConverter>();
        app.init_resource::<Points>();
        app.init_resource::<MaxCoordinates>();
        app.init_state::<ModelState>();
        app.add_systems(
            Update,
            (update_points, update_max_coordinates, draw_points)
                .chain()
                .run_if(in_state(ModelState::Updating)),
        );
    }
}
