pub mod setup {

    use bevy::prelude::*;
    use crate::parameters::Parameters;
    use crate::graph::*;

    pub struct SetupPlugin;

    impl Plugin for SetupPlugin {
        fn build(&self, app: &mut App) {
            app.init_resource::<Parameters>();
            app.add_systems(Startup, (setup, set_grid, set_points, draw_grid, draw_points).chain());
        }
    }

    fn setup(mut commands: Commands) {
        commands.spawn(Camera2d);
    }
}