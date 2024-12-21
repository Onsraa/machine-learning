pub mod points {

    use bevy::prelude::*;
    use crate::parameters::Parameters;
    use crate::algorithms::*;

    #[derive(Resource)]
    pub struct Points(pub [[f32; 2]; 3]);

    pub fn set_points(mut commands: Commands) {
        let points: [[f32; 2]; 3] = [
            [1.0,1.0],
            [2.0,3.0],
            [3.0,3.0],
        ];
        commands.insert_resource(Points(points));
    }

    pub fn draw_points(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
        points: Res<Points>,
        parameters: Res<Parameters>,
    ) {
        let ((min_x, max_x), (min_y, max_y)) = min_max_mat2(&points.0);

        let drawing_width = parameters.width - 2.0 * parameters.padding;
        let drawing_height = parameters.height - 2.0 * parameters.padding;

        let scale_x = drawing_width / (max_x - min_x);
        let scale_y = drawing_height / (max_y - min_y);

        let scale = scale_x.min(scale_y);

        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        for &point in points.0.iter() {
            let tx = point[0] - center_x;
            let ty = point[1] - center_y;

            let sx = tx * scale;
            let sy = ty * scale;

            commands.spawn((
                Mesh2d(meshes.add(Circle::new(parameters.points_size))),
                MeshMaterial2d(materials.add(Color::BLACK)),
                Transform::from_xyz(sx, sy, 1.0),
            ));
        }
    }
}