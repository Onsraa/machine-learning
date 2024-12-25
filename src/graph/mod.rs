mod points;
mod grid;
mod egui;

pub use grid::grid::{set_grid, draw_grid};
pub use points::points::draw_points;
pub use egui::egui::draw_plot;