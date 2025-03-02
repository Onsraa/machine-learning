mod algorithms;
mod components;
mod data;
mod params;
mod plugins;
mod resources;
mod states;
mod systems;
mod ui;

use crate::data::ModelState;
use crate::plugins::models::ModelsPlugin;
use crate::plugins::setup::SetupPlugin;
use crate::resources::game_image_state::GameImageState;
use crate::resources::mlp_config::MLPConfig;
use crate::resources::mlp_image_config::MLPImageConfig;
use crate::resources::model_managers::ModelManager;
use crate::resources::training::TrainingState;
use crate::states::{AppState, TrainingState as AppTrainingState};
use crate::systems::image_training::image_training_system;
use crate::systems::navigation::handle_navigation;
use crate::systems::training::training_system;
use crate::ui::game_classifier::game_classifier_ui;
use crate::ui::menu::main_menu_ui;
use crate::ui::mlp_config::mlp_config_ui;
use crate::ui::model_manager_ui::model_manager_ui;
use crate::ui::model_selector::update_model_selector_ui;
use crate::ui::models::update_test_case_ui;
use crate::ui::rbf_config::rbf_config_ui;
use crate::ui::svm_config::svm_config_ui;
use crate::ui::training::training_ui_system;
use bevy::{color::palettes::css::*, prelude::*};
use bevy_egui::EguiPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::Srgba(WHITE_SMOKE)))
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)

        // États de l'application
        .init_state::<AppState>()
        .init_state::<AppTrainingState>()
        .init_state::<ModelState>()

        // Ressources globales
        .init_resource::<TrainingState>()
        .init_resource::<ModelManager>()

        // Ressources pour les cas de tests
        .init_resource::<MLPConfig>()

        // Ressources pour la classification d'images
        .init_resource::<GameImageState>()
        .init_resource::<MLPImageConfig>()

        // Plugins pour les cas de test
        .add_plugins(SetupPlugin)
        .add_plugins(ModelsPlugin)

        // Systèmes par état - MENU
        .add_systems(Update, main_menu_ui.run_if(in_state(AppState::Menu)))

        // Systèmes par état - CAS TESTS
        .add_systems(
            Update,
            (
                update_test_case_ui,
                update_model_selector_ui,
                mlp_config_ui,
                rbf_config_ui,
                svm_config_ui,
                model_manager_ui,
                training_ui_system,
                training_system,
            )
                .run_if(in_state(AppState::CasTests))
        )

        // Systèmes par état - CLASSIFICATION JEUX
        .add_systems(Update, game_classifier_ui.run_if(in_state(AppState::ClassificationJeux)))
        .add_systems(Update, image_training_system.run_if(in_state(AppState::ClassificationJeux)))

        // Système de navigation global
        .add_systems(Update, handle_navigation)

        .run();
}