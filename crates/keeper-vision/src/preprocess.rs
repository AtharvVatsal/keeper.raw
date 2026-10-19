use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{DynamicImage, GrayImage, Rgb, RgbImage};
use keeper_core::types::BoundingBox;
use ndarray::Array4;
use tracing::debug;

pub fn decode_jpeg(jpeg_bytes: &[u8]) -> Result<DynamicImage> {
    let img = image::load_from_memory(jpeg_bytes).context("Failed to decode JPEG preview image")?;
    Ok(img)
}

// =============================================================================
// Decode-once analysis image
//
// Embedded RAW previews are often full-size (24 MP+). Decoding and processing
// them at full size in every stage made the pipeline ~2.7 s/image. Instead we
// decode each preview ONCE into two sizes and share them across all stages:
//   - `work`  : 1/2 scale for large previews — keeps eye detail for focus/blink
//   - `small` : ≤ 640 px long edge — all that pHash and YOLO (640×640) need
// =============================================================================

/// Longest edge (px) of the `small` copy. YOLO's input is 640×640, so a larger
/// copy would only be shrunk again.
pub const SMALL_LONG_EDGE: u32 = 640;

/// Previews with a long edge of at least this many pixels are decoded at
/// 1/2 scale (e.g. 6000×4000 → 3000×2000); smaller ones at full size.
pub const HALF_SCALE_MIN_LONG_EDGE: u32 = 4000;

/// One preview, decoded once and shared by every analysis stage.
pub struct AnalysisImage {
    /// Size of the original embedded preview. Bounding boxes stored in
    /// `ImageScore` use this coordinate space (the loupe zooms on it).
    pub preview_width: u32,
    pub preview_height: u32,
    /// Working image for focus scoring and blink detection.
    pub work: DynamicImage,
    /// Small copy (≤ `SMALL_LONG_EDGE`) for pHash and face detection.
    pub small: DynamicImage,
}

impl AnalysisImage {
    /// Multiply `work` coordinates by this to get preview coordinates.
    pub fn work_to_preview(&self) -> f32 {
        self.preview_width as f32 / self.work.width() as f32
    }

    /// Multiply `small` coordinates by this to get preview coordinates.
    pub fn small_to_preview(&self) -> f32 {
        self.preview_width as f32 / self.small.width() as f32
    }
}

/// Decodes a JPEG preview once into an [`AnalysisImage`].
///
/// Uses `jpeg-decoder`'s IDCT scaling for a cheap 1/2-scale decode; falls back
/// to a full decode via the `image` crate for anything it can't handle
/// (e.g. CMYK or 16-bit JPEGs).
///
/// `orientation` is the shot's EXIF orientation (1–8). The image is rotated
/// upright before anything else, so faces are analysed the right way up and
/// all sizes/coordinates in the result are upright.
pub fn decode_for_analysis(jpeg_bytes: &[u8], orientation: u8) -> Result<AnalysisImage> {
    let (mut preview_width, mut preview_height, mut work) = match decode_scaled(jpeg_bytes) {
        Some(decoded) => decoded,
        None => {
            let img = decode_jpeg(jpeg_bytes)?;
            (img.width(), img.height(), img)
        }
    };

    // ---- Rotate upright ----------------------------------------------------
    // Portrait shots are stored sideways; analysing them as-is shows faces
    // lying on their side to the face and landmark models.
    if let Some(o) = Orientation::from_exif(orientation) {
        work.apply_orientation(o);
        if (5..=8).contains(&orientation) {
            std::mem::swap(&mut preview_width, &mut preview_height);
        }
    }

    let small = if work.width().max(work.height()) > SMALL_LONG_EDGE {
        // `thumbnail` is a fast area-averaging downscale — ~4× cheaper than
        // a Triangle-filter resize on these sizes.
        work.thumbnail(SMALL_LONG_EDGE, SMALL_LONG_EDGE)
    } else {
        work.clone()
    };

    debug!(
        "Decoded preview {}x{} → work {}x{}, small {}x{}",
        preview_width,
        preview_height,
        work.width(),
        work.height(),
        small.width(),
        small.height()
    );

    Ok(AnalysisImage {
        preview_width,
        preview_height,
        work,
        small,
    })
}

/// Returns `(preview_width, preview_height, decoded image)`, or `None` if
/// `jpeg-decoder` can't produce an 8-bit RGB or greyscale image.
fn decode_scaled(jpeg_bytes: &[u8]) -> Option<(u32, u32, DynamicImage)> {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_bytes);
    decoder.read_info().ok()?;
    let info = decoder.info()?;
    let (preview_width, preview_height) = (info.width as u32, info.height as u32);

    let (out_w, out_h) = if preview_width.max(preview_height) >= HALF_SCALE_MIN_LONG_EDGE {
        // `scale` picks the smallest of 1/8, 1/4, 1/2, 1 that is ≥ the request.
        decoder.scale(info.width / 2, info.height / 2).ok()?
    } else {
        (info.width, info.height)
    };

    let pixels = decoder.decode().ok()?;
    let (w, h) = (out_w as u32, out_h as u32);

    let image = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => {
            DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, pixels)?)
        }
        jpeg_decoder::PixelFormat::L8 => {
            DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, pixels)?)
        }
        _ => return None,
    };

    Some((preview_width, preview_height, image))
}

/// Reads the EXIF orientation (1–8) embedded in a JPEG file itself, for
/// inputs that are plain JPEGs rather than RAW previews. Returns 1 (upright)
/// if there is no tag or it can't be read.
pub fn jpeg_orientation(jpeg_bytes: &[u8]) -> u8 {
    use image::ImageDecoder;
    image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg_bytes))
        .and_then(|mut decoder| decoder.orientation())
        .map(|o| o.to_exif())
        .unwrap_or(1)
}

/// Samples a `size`×`size` RGB patch from `img`: a square of side `side`
/// pixels centred on (`cx`, `cy`), rotated by `angle` radians (positive =
/// clockwise in image coordinates, so passing a face's tilt levels it).
/// Bilinear interpolation; areas outside the image are black.
///
/// This is how face crops are prepared for the landmark model: square (so
/// vertical eye distances aren't squashed) and level (so tilted heads look
/// like the upright faces the model was trained on).
pub fn sample_square(
    img: &DynamicImage,
    cx: f32,
    cy: f32,
    side: f32,
    angle: f32,
    size: u32,
) -> RgbImage {
    let owned;
    let rgb = match img.as_rgb8() {
        Some(rgb) => rgb,
        None => {
            owned = img.to_rgb8();
            &owned
        }
    };
    let (sin, cos) = angle.sin_cos();
    let half = size as f32 / 2.0;
    let (max_x, max_y) = (rgb.width() as f32 - 1.0, rgb.height() as f32 - 1.0);

    RgbImage::from_fn(size, size, |u, v| {
        // Offset from the patch centre, in source pixels, then rotate.
        let dx = (u as f32 + 0.5 - half) / size as f32 * side;
        let dy = (v as f32 + 0.5 - half) / size as f32 * side;
        let x = cx + dx * cos - dy * sin;
        let y = cy + dx * sin + dy * cos;
        if x < 0.0 || y < 0.0 || x >= max_x || y >= max_y {
            return Rgb([0, 0, 0]);
        }

        let (x0, y0) = (x as u32, y as u32);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let (p00, p10) = (rgb.get_pixel(x0, y0).0, rgb.get_pixel(x0 + 1, y0).0);
        let (p01, p11) = (rgb.get_pixel(x0, y0 + 1).0, rgb.get_pixel(x0 + 1, y0 + 1).0);
        let mut out = [0u8; 3];
        for c in 0..3 {
            let value = p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
                + p10[c] as f32 * fx * (1.0 - fy)
                + p01[c] as f32 * (1.0 - fx) * fy
                + p11[c] as f32 * fx * fy;
            out[c] = value.round().clamp(0.0, 255.0) as u8;
        }
        Rgb(out)
    })
}

/// Scales a bounding box by `factor` (e.g. from `small` to preview coordinates).
pub fn scale_bbox(bbox: &BoundingBox, factor: f32) -> BoundingBox {
    BoundingBox {
        x: bbox.x * factor,
        y: bbox.y * factor,
        width: bbox.width * factor,
        height: bbox.height * factor,
    }
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

    /// Maps a single point from model-input coordinates to the original image.
    pub fn to_original_point(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)
    }
}

pub fn image_to_yolo_tensor(img: &DynamicImage) -> Array4<f32> {
    image_to_nchw_tensor(img, SIMPLE_MEAN, SIMPLE_STD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;

    fn jpeg_of_size(w: u32, h: u32) -> Vec<u8> {
        let img = RgbImage::from_fn(w, h, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, 90)
            .encode_image(&img)
            .unwrap();
        bytes
    }

    #[test]
    fn test_large_preview_decodes_at_half_scale() {
        let decoded = decode_for_analysis(&jpeg_of_size(4800, 3200), 1).unwrap();
        assert_eq!(
            (decoded.preview_width, decoded.preview_height),
            (4800, 3200)
        );
        assert_eq!((decoded.work.width(), decoded.work.height()), (2400, 1600));
        assert_eq!(
            decoded.small.width().max(decoded.small.height()),
            SMALL_LONG_EDGE
        );
        assert!((decoded.work_to_preview() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_small_preview_decodes_at_full_size() {
        let decoded = decode_for_analysis(&jpeg_of_size(1600, 1067), 1).unwrap();
        assert_eq!((decoded.work.width(), decoded.work.height()), (1600, 1067));
        assert!((decoded.work_to_preview() - 1.0).abs() < 1e-6);
        assert!(decoded.small.width() <= SMALL_LONG_EDGE);
    }

    #[test]
    fn test_tiny_preview_is_not_upscaled() {
        let decoded = decode_for_analysis(&jpeg_of_size(320, 240), 1).unwrap();
        assert_eq!((decoded.small.width(), decoded.small.height()), (320, 240));
    }

    #[test]
    fn test_rotated_shot_is_analysed_upright() {
        // Stored sideways: left half red, right half blue.
        let img = RgbImage::from_fn(4800, 3200, |x, _| {
            if x < 2400 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        });
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, 90)
            .encode_image(&img)
            .unwrap();

        // EXIF 6 = rotate 90° clockwise → the left (red) half ends up on top.
        let decoded = decode_for_analysis(&bytes, 6).unwrap();
        assert_eq!(
            (decoded.preview_width, decoded.preview_height),
            (3200, 4800)
        );
        assert_eq!((decoded.work.width(), decoded.work.height()), (1600, 2400));
        assert!(decoded.small.height() > decoded.small.width());

        let work = decoded.work.to_rgb8();
        let top = work.get_pixel(800, 100);
        let bottom = work.get_pixel(800, 2300);
        assert!(
            top[0] > 200 && top[2] < 60,
            "top should be red, got {top:?}"
        );
        assert!(
            bottom[2] > 200 && bottom[0] < 60,
            "bottom should be blue, got {bottom:?}"
        );
    }

    #[test]
    fn test_sample_square_unrotated_and_rotated() {
        // 100×100: left half red, right half blue.
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(100, 100, |x, _| {
            if x < 50 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        }));

        // No rotation: left of patch red, right blue.
        let p = sample_square(&img, 50.0, 50.0, 60.0, 0.0, 20);
        assert_eq!(p.get_pixel(2, 10).0, [255, 0, 0]);
        assert_eq!(p.get_pixel(17, 10).0, [0, 0, 255]);

        // Rotated 90°: the patch's x axis now points down the source image,
        // so its top rows sample the left (red) half.
        let p = sample_square(&img, 50.0, 50.0, 60.0, std::f32::consts::FRAC_PI_2, 20);
        assert_eq!(p.get_pixel(10, 17).0, [255, 0, 0]);
        assert_eq!(p.get_pixel(10, 2).0, [0, 0, 255]);

        // Outside the image → black.
        let p = sample_square(&img, 0.0, 0.0, 60.0, 0.0, 20);
        assert_eq!(p.get_pixel(0, 0).0, [0, 0, 0]);
    }

    #[test]
    fn test_jpeg_orientation_defaults_to_upright() {
        assert_eq!(jpeg_orientation(&jpeg_of_size(64, 48)), 1);
        assert_eq!(jpeg_orientation(b"not a jpeg"), 1);
    }

    #[test]
    fn test_scale_bbox() {
        let b = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };
        let s = scale_bbox(&b, 2.0);
        assert_eq!((s.x, s.y, s.width, s.height), (20.0, 40.0, 60.0, 80.0));
    }
}
