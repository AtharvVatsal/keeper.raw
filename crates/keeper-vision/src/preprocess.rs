use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array4;
use tracing::debug;

pub fn decode_jpeg(jpeg_bytes: &[u8]) -> Result<DynamicImage> {
    let img = image::load_from_memory(jpeg_bytes).context("Failed to decode JPEG preview image")?;
    Ok(img)
}

pub fn resize_image(img: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    debug!(
        "Resizing image from {}x{} to {}x{}",
        img.width(),
        img.height(),
        width,
        height
    );
    img.resize_exact(width, height, FilterType::Triangle)
}

pub fn image_to_nchw_tensor(img: &DynamicImage, mean: [f32; 3], std_dev: [f32; 3]) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = (pixel.0[c] as f32 / 255.0 - mean[c]) / std_dev[c];
            }
        }
    }

    tensor
}

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub const SIMPLE_MEAN: [f32; 3] = [0.0, 0.0, 0.0];
pub const SIMPLE_STD: [f32; 3] = [1.0, 1.0, 1.0];

pub fn letterbox(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> (DynamicImage, LetterboxTransform) {
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
    let (tgt_w, tgt_h) = (target_width as f32, target_height as f32);

    let scale = (tgt_w / orig_w).min(tgt_h / orig_h);

    let new_w = (orig_w * scale).round() as u32;
    let new_h = (orig_h * scale).round() as u32;

    let pad_x = (target_width - new_w) / 2;
    let pad_y = (target_height - new_h) / 2;

    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    let gray = Rgb([114u8, 114, 114]);
    let mut canvas = RgbImage::from_pixel(target_width, target_height, gray);

    let resized_rgb = resized.to_rgb8();
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized_rgb.get_pixel(x, y);
            canvas.put_pixel(x + pad_x, y + pad_y, *pixel);
        }
    }

    let transform = LetterboxTransform {
        scale,
        pad_x: pad_x as f32,
        pad_y: pad_y as f32,
        orig_width: orig_w,
        orig_height: orig_h,
    };

    (DynamicImage::ImageRgb8(canvas), transform)
}

#[derive(Debug, Clone, Copy)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub orig_width: f32,
    pub orig_height: f32,
}

impl LetterboxTransform {
    pub fn to_original_coords(&self, cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        let orig_cx = (cx - self.pad_x) / self.scale;
        let orig_cy = (cy - self.pad_y) / self.scale;
        let orig_w = w / self.scale;
        let orig_h = h / self.scale;

        let x1 = (orig_cx - orig_w / 2.0).max(0.0);
        let y1 = (orig_cy - orig_h / 2.0).max(0.0);
        let x2 = (orig_cx + orig_w / 2.0).min(self.orig_width);
        let y2 = (orig_cy + orig_h / 2.0).min(self.orig_height);

        (x1, y1, x2 - x1, y2 - y1)
    }
}

pub fn image_to_yolo_tensor(img: &DynamicImage) -> Array4<f32> {
    image_to_nchw_tensor(img, SIMPLE_MEAN, SIMPLE_STD)
}
