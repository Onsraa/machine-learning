use crate::states::AppState;
use bevy::prelude::*;

pub fn handle_navigation(
    keyboard_events: Res<ButtonInput<KeyCode>>,
    current_state: Res<State<AppState>>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if keyboard_events.just_pressed(KeyCode::Escape) {
        if *current_state.get() != AppState::Menu {
            next_state.set(AppState::Menu);
        }
    }
}