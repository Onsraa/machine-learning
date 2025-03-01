use crate::resources::mlp_config::MLPConfig;
use crate::resources::model_managers::ModelManager;
use crate::resources::training::TrainingState;
use crate::systems::training::training_system;
use crate::ui::mlp_config::mlp_config_ui;
use crate::ui::model_manager_ui::model_manager_ui;
use crate::ui::model_selector::update_model_selector_ui;
use crate::ui::models::update_test_case_ui;
use crate::ui::rbf_config::rbf_config_ui;
use crate::ui::svm_config::svm_config_ui;
use crate::ui::training::training_ui_system;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin);
        app.init_resource::<TrainingState>();
        app.init_resource::<MLPConfig>();
        app.init_resource::<ModelManager>();
        app.add_systems(Update, training_ui_system);
        app.add_systems(Update, training_system);
        app.add_systems(Update, update_model_selector_ui);
        app.add_systems(Update, mlp_config_ui);
        app.add_systems(Update, update_test_case_ui);
        app.add_systems(Update, rbf_config_ui);
        app.add_systems(Update, svm_config_ui);
        app.add_systems(Update, model_manager_ui);
    }
}
