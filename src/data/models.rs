use bevy::color::palettes::css as color;
use bevy::prelude::*;
use rand::Rng;

pub struct Point(pub f64, pub f64, pub f64, pub Color); // x, y, z, color

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

pub fn set_points(mut points: ResMut<Points>, data_model: Res<DataModel>) {
    points.0 = match *data_model {
        DataModel::LinearSimple => create_linear_simple_model(),
        DataModel::LinearMultiple => create_linear_multiple_model(),
        DataModel::XOR => create_xor_model(),
        DataModel::Cross => create_cross_model(),
        DataModel::MultiLinear3Classes => create_multi_linear_3_classes_model(),
        DataModel::MultiCross => create_multi_cross_model(),
        DataModel::LinearSimple2d => create_linear_simple_2d_model(),
        DataModel::LinearSimple3d => create_linear_simple_3d_model(),
        DataModel::LinearTricky3d => create_linear_tricky_3d_model(),
        DataModel::NonLinearSimple2d => create_non_linear_simple_2d_model(),
        DataModel::NonLinearSimple3d => create_non_linear_simple_3d_model(),
    }
}

fn create_linear_simple_model() -> Vec<Point> {
    vec![
        Point(1.0, 1.0, 0.0, Color::from(color::BLUE)),
        Point(2.0, 3.0, 0.0, Color::from(color::RED)),
        Point(3.0, 3.0, 0.0, Color::from(color::RED)),
    ]
}

fn create_linear_multiple_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(100);

    for _ in 0..50 {
        let x = rng.gen::<f64>() * 0.9 + 1.0;
        let y = rng.gen::<f64>() * 0.9 + 1.0;
        pts.push(Point(x, y, 0.0, Color::from(color::BLUE)));
    }

    for _ in 0..50 {
        let x = rng.gen::<f64>() * 0.9 + 2.0;
        let y = rng.gen::<f64>() * 0.9 + 2.0;
        pts.push(Point(x, y, 0.0, Color::from(color::RED)));
    }

    pts
}

fn create_xor_model() -> Vec<Point> {
    vec![
        Point(1.0, 0.0, 0.0, Color::from(color::BLUE)),
        Point(0.0, 1.0, 0.0, Color::from(color::BLUE)),
        Point(0.0, 0.0, 0.0, Color::from(color::RED)),
        Point(1.0, 1.0, 0.0, Color::from(color::RED)),
    ]
}

fn create_cross_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(500);

    for _ in 0..500 {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        let label = if x.abs() <= 0.3 || y.abs() <= 0.3 {
            1
        } else {
            -1
        };
        let color = if label == 1 {
            Color::from(color::BLUE)
        } else {
            Color::from(color::RED)
        };
        pts.push(Point(x, y, 0.0, color));
    }
    pts
}

fn create_multi_linear_3_classes_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::new();

    for _ in 0..500 {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);

        let c1 = -x - y - 0.5 > 0.0 && y < 0.0 && (x - y - 0.5) < 0.0;
        let c2 = -x - y - 0.5 < 0.0 && y > 0.0 && (x - y - 0.5) < 0.0;
        let c3 = -x - y - 0.5 < 0.0 && y < 0.0 && (x - y - 0.5) > 0.0;

        let color = if c1 {
            Some(Color::from(color::BLUE))
        } else if c2 {
            Some(Color::from(color::RED))
        } else if c3 {
            Some(Color::from(color::GREEN))
        } else {
            None
        };

        if let Some(col) = color {
            pts.push(Point(x, y, 0.0, col));
        }
    }

    pts
}

fn create_multi_cross_model() -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(1000);

    for _ in 0..1000 {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);

        let cond_blue: f64 = x % 0.5;
        let cond_red: f64 = y % 0.5;

        let cond_blue_ok = cond_blue.abs() <= 0.25 && cond_red.abs() > 0.25;
        let cond_red_ok = cond_blue.abs() > 0.25 && cond_red.abs() <= 0.25;

        let color = if cond_blue_ok {
            Color::from(color::BLUE)
        } else if cond_red_ok {
            Color::from(color::RED)
        } else {
            Color::from(color::GREEN)
        };

        pts.push(Point(x, y, 0.0, color));
    }

    pts
}

fn create_linear_simple_2d_model() -> Vec<Point> {
    vec![
        Point(1.0, 2.0, 0.0, Color::BLACK),
        Point(2.0, 3.0, 0.0, Color::BLACK),
    ]
}

fn create_linear_simple_3d_model() -> Vec<Point> {
    vec![
        Point(1.0, 1.0, 2.0, Color::from(color::BLACK)),
        Point(2.0, 2.0, 3.0, Color::from(color::BLACK)),
        Point(3.0, 1.0, 2.5, Color::from(color::BLACK)),
    ]
}

fn create_linear_tricky_3d_model() -> Vec<Point> {
    vec![
        Point(1.0, 1.0, 1.0, Color::from(color::BLACK)),
        Point(2.0, 2.0, 2.0, Color::from(color::BLACK)),
        Point(3.0, 3.0, 3.0, Color::from(color::BLACK)),
    ]
}

fn create_non_linear_simple_2d_model() -> Vec<Point> {
    vec![
        Point(1.0, 2.0, 0.0, Color::BLACK),
        Point(2.0, 3.0, 0.0, Color::BLACK),
        Point(3.0, 2.5, 0.0, Color::BLACK),
    ]
}

fn create_non_linear_simple_3d_model() -> Vec<Point> {
    vec![
        Point(1.0, 0.0, 2.0, Color::from(color::BLACK)),
        Point(0.0, 1.0, 1.0, Color::from(color::BLACK)),
        Point(1.0, 1.0, -2.0, Color::from(color::BLACK)),
        Point(0.0, 0.0, -1.0, Color::from(color::BLACK)),
    ]
}
