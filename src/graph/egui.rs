pub mod egui {
    use std::collections::HashMap;
    use bevy::prelude::*;
    use bevy_egui::{
        egui::{
            self,
        },
        EguiContexts,
    };
    use egui_plot::{Plot, PlotPoints};
    use egui::Color32;
    use crate::data::Points;

    pub fn draw_plot(
        mut contexts: EguiContexts,
        points: Res<Points>,
    ) {
        let ctx = contexts.ctx_mut();

        egui::Window::new("Data plot")
            .movable(false)
            .show(ctx, |ui| {

                let mut hashmap_points : HashMap<Color32, Vec<[f64; 2]>> = HashMap::new();

                for point in points.0.iter() {
                    let xy: [f64;2] = [point.0, point.1];
                    let color: Color32 = bevy_color_to_egui(point.3);
                    hashmap_points.entry(color).or_default().push(xy);
                }

                Plot::new("Data").view_aspect(2.0).show(ui, |plot_ui| {
                    for (color, points) in hashmap_points {
                        plot_ui.points(egui_plot::Points::new(PlotPoints::new(points)).color(color));
                    }
                });
            });
    }

    fn bevy_color_to_egui(c: Color) -> Color32 {

        let (mut r, mut g, mut b) = (1.0, 1.0, 1.0);

        if let Color::Srgba(srgba) = c {
            r = srgba.red;
            g = srgba.green;
            b = srgba.blue;
        }
        Color32::from_rgba_unmultiplied(
            (r * 255.) as u8,
            (g * 255.) as u8,
            (b * 255.) as u8,
            255,
        )
    }
}