pub mod grid {

    use bevy::prelude::*;
    use crate::parameters::Parameters;

    #[derive(Resource)]
    pub struct Grid(pub f64, f64);

    pub fn set_grid(mut commands: Commands, parameters: Res<Parameters>) {
        commands.insert_resource(Grid(parameters.width, parameters.height));
    }

    pub fn draw_grid<ColorMaterial>(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
        grid: Res<Grid>
    ) {
        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(grid.0 as f32, grid.1 as f32))),
            MeshMaterial2d(materials.add(Color::WHITE)),
            Transform::default()
        )
        );
    }
}