use crate::algorithms::model_linear::*;
use bevy::prelude::*;

pub struct LinearModelPlugin;

impl Plugin for LinearModelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LinearModel>();
        app.init_resource::<TrainingParameters>();
        app.init_resource::<TrainingMetrics>();
        app.init_state::<TrainingState>();
        app.add_systems(Startup, setup_model);
        app.add_systems(
            Update,
            (
                training_ui,
                train_epoch_system.run_if(in_state(TrainingState::Running)),
            ),
        );
    }
}
