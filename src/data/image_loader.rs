use crate::data::image_processing::{ImagePreprocessor, TARGET_SIZE};
use bevy::prelude::*;
use image::{GrayImage, RgbaImage};
use nalgebra::DVector;
use std::path::Path;

pub struct ImageLoader;

impl ImageLoader {
    pub fn load_and_preprocess<P: AsRef<Path>>(path: P) -> Result<DVector<f64>, String> {
        ImagePreprocessor::load_and_preprocess(path)
            .map_err(|e| format!("Erreur lors du prétraitement de l'image: {}", e))
    }

    pub fn create_preview_texture(
        img_vec: &DVector<f64>,
        textures: &mut Assets<Image>
    ) -> Handle<Image> {
        // Reconstruire l'image à partir du vecteur
        let mut img_buffer = GrayImage::new(TARGET_SIZE.0, TARGET_SIZE.1);

        for (i, &pixel_value) in img_vec.iter().enumerate() {
            let x = (i as u32) % TARGET_SIZE.0;
            let y = (i as u32) / TARGET_SIZE.0;
            let pixel = image::Luma([(pixel_value * 255.0) as u8]);

            if x < TARGET_SIZE.0 && y < TARGET_SIZE.1 {
                img_buffer.put_pixel(x, y, pixel);
            }
        }

        let rgba_buffer = RgbaImage::from_fn(TARGET_SIZE.0, TARGET_SIZE.1, |x, y| {
            let gray = img_buffer.get_pixel(x, y)[0];
            image::Rgba([gray, gray, gray, 255])
        });

        let data = rgba_buffer.into_raw();

        let mut bevy_image = Image::new(
            bevy::render::render_resource::Extent3d {
                width: TARGET_SIZE.0,
                height: TARGET_SIZE.1,
                depth_or_array_layers: 1,
            },
            bevy::render::render_resource::TextureDimension::D2,
            data,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            bevy::render::render_asset::RenderAssetUsages::default(),
        );

        bevy_image.texture_descriptor.usage = bevy::render::render_resource::TextureUsages::TEXTURE_BINDING
            | bevy::render::render_resource::TextureUsages::COPY_DST;

        textures.add(bevy_image)
    }
}