pub mod graph {

    use bevy::prelude::Resource;
    const WIDTH:f64 = 500.0;
    const HEIGHT:f64 = 500.0;
    const POINT_SIZE:f64 = 2.5;
    const PADDING:f64 = 20.0;
    const NUMBER_TICKS: u8 = 5;

    #[derive(Resource)]
    pub struct Parameters {
        pub width: f64,
        pub height: f64,
        pub points_size: f64,
        pub padding: f64,
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