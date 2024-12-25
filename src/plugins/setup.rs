pub mod setup {

    use bevy::prelude::*;
    use crate::data::*;
    use crate::parameters::Parameters;
    use crate::graph::*;

    pub struct SetupPlugin;

    impl Plugin for SetupPlugin {
        fn build(&self, app: &mut App) {
            app.init_resource::<Parameters>();
            // app.add_systems(Startup, (setup, set_grid, draw_grid, set_data_model, set_points, draw_points).chain());
            app.add_systems(Startup, (setup, set_data_model, set_points).chain());
            app.add_systems(Update, draw_plot);
        }
    }

    fn setup(mut commands: Commands) {
        commands.spawn(Camera2d);
    }
}