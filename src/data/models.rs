pub mod models {

    use bevy::prelude::*;
    use bevy::color::palettes::css as color;

    pub struct Point(pub f32, pub f32, pub f32, pub Color); // x, y, y, color

    #[derive(Resource, Default)]
    pub struct Points(pub Vec<Point>);

    #[derive(Resource, Default)]
    pub enum DataModel {
        #[default]
        LinearSimple,
        LinearMultiple,
        XOR,
        Cross,
        MultiLinear3Classes,
        MultiCross,
        LinearSimple2d,
        LinearSimple3d,
        LinearTricky3d,
        NonLinearSimple2d,
        NonLinearSimple3d,
    }

    pub fn set_data_model(mut commands: Commands) {
        commands.init_resource::<DataModel>();
        commands.init_resource::<Points>();
    }

    pub fn set_points(mut points: ResMut<Points>, data_model: Res<DataModel>) {
        points.0 = match data_model.into_inner() {
            DataModel::LinearSimple => create_linear_simple_model(),
            _ => vec![],
        }
    }

    fn create_linear_simple_model() -> Vec<Point> {
        vec![
            Point(1.0, 1.0, 1.0, Color::from(color::BLUE)),
            Point(2.0, 3.0, 1.0, Color::from(color::RED)),
            Point(3.0, 3.0, 1.0, Color::from(color::RED)),
        ]
    }
}