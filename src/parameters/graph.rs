pub mod graph {

    use bevy::prelude::Resource;
    const WIDTH:f32 = 500.0;
    const HEIGHT:f32 = 500.0;
    const POINT_SIZE:f32 = 2.5;
    const PADDING:f32 = 20.0;
    const NUMBER_TICKS: u8 = 5;

    #[derive(Resource)]
    pub struct Parameters {
        pub width: f32,
        pub height: f32,
        pub points_size: f32,
        pub padding: f32,
        pub number_ticks: u8,
    }

    impl Default for Parameters {
        fn default() -> Self {
            Self {
                width: WIDTH,
                height: HEIGHT,
                points_size: POINT_SIZE,
                padding: PADDING,
                number_ticks: NUMBER_TICKS,
            }
        }
    }
}