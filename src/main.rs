use bevy::prelude::*;
// use nalgebra::*;

const WIDTH:f32 = 300.0;
const HEIGHT:f32 = 300.0;
const POINT_SIZE:f32 = 2.5;
const PADDING:f32 = 20.0;

#[derive(Resource)]
struct Grid(f32, f32);

#[derive(Resource)]
struct Points([[f32; 2]; 3]);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SetupPlugin)
        .run();
}

struct SetupPlugin;

impl Plugin for SetupPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup, set_grid, set_points).chain());
    }
}

fn setup(mut commands: Commands) {
    let points: [[f32; 2]; 3] = [
        [1.0,1.0],
        [2.0,3.0],
        [3.0,3.0],
    ];
    commands.insert_resource(Points(points));
    commands.insert_resource(Grid(WIDTH, HEIGHT));
    commands.spawn(Camera2d);
}

fn min_max(points: &[[f32; 2]]) -> ((f32, f32), (f32, f32)) {
    points.iter().fold(
        ((f32::INFINITY, f32::NEG_INFINITY), (f32::INFINITY, f32::NEG_INFINITY)),
        |((min_x, max_x), (min_y, max_y)), &[x, y]| {
            (
                (min_x.min(x), max_x.max(x)),
                (min_y.min(y), max_y.max(y)),
            )
        },
    )
}

fn set_grid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    grid: Res<Grid>
) {
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(grid.0, grid.1))),
        MeshMaterial2d(materials.add(Color::WHITE)),
        Transform::default()
    )
    );
}

fn set_points(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    points: Res<Points>,
) {
    let ((min_x, max_x), (min_y, max_y)) = min_max(&points.0);

    let drawing_width = WIDTH - 2.0 * PADDING;
    let drawing_height = HEIGHT - 2.0 * PADDING;

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
            Mesh2d(meshes.add(Circle::new(POINT_SIZE))),
            MeshMaterial2d(materials.add(Color::BLACK)),
            Transform::from_xyz(sx, sy, 1.0),
        ));
    }
}