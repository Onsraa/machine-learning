pub mod graph {

    pub fn min_max_mat2(points: &[[f32; 2]]) -> ((f32, f32), (f32, f32)) {
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

    pub fn min_max_mat3(points: &[[f32; 3]]) -> ((f32, f32), (f32, f32), (f32, f32)) {
        points.iter().fold(
            ((f32::INFINITY, f32::NEG_INFINITY), (f32::INFINITY, f32::NEG_INFINITY), (f32::INFINITY, f32::NEG_INFINITY)),
            |((min_x, max_x), (min_y, max_y), (min_z, max_z)), &[x, y, z]| {
                (
                    (min_x.min(x), max_x.max(x)),
                    (min_y.min(y), max_y.max(y)),
                    (min_y.min(z), max_y.max(z)),
                )
            },
        )
    }
}