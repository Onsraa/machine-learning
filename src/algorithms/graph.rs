pub mod graph {

    use crate::data::Points;

    pub fn min_max(points: &Points) -> ((f32, f32), (f32, f32), (f32, f32)) {
        points.0.iter().fold(
            (
                (f32::INFINITY, f32::NEG_INFINITY), // min_x, max_x
                (f32::INFINITY, f32::NEG_INFINITY), // min_y, max_y
                (f32::INFINITY, f32::NEG_INFINITY), // min_z, max_z
            ),
            |((min_x, max_x), (min_y, max_y), (min_z, max_z)), point| {
                (
                    (min_x.min(point.0), max_x.max(point.0)),
                    (min_y.min(point.1), max_y.max(point.1)),
                    (min_z.min(point.2), max_z.max(point.2)),
                )
            },
        )
    }
}