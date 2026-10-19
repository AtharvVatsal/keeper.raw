//! Frame-level quality checks that don't depend on faces or focus.
//!
//! Unusable frames — black (lens cap, misfire) or blank (featureless wall) —
//! must be rejected, but neither brightness nor sharpness can tell them apart
//! from legitimate shots:
//! - concert photos average 9–15 / 255 brightness yet are fine (bright
//!   performer on a dark stage);
//! - a small bird in a clear sky has *lower* frame-wide contrast and
//!   sharpness than a blank grey wall.
//!
//! What separates them is **local** contrast: any real subject, however
//! small, creates at least a few 8×8 patches with strong contrast; a blank or
//! black frame has none anywhere.
//!
//! Measured on 118 real NEFs (strongest local contrast, 5th-highest block):
//! the two junk frames scored 4.0 and 9.4; the lowest legitimate shot 126.9;
//! bird-in-sky shots 153–184.

use image::DynamicImage;

/// Frames whose strongest local contrast is below this (0–255 luminance
/// levels) are unusable. ~4× above the junk frames measured, ~3× below the
/// lowest real photo.
pub const UNUSABLE_LOCAL_CONTRAST: f32 = 40.0;

/// Side of the square blocks local contrast is measured in.
const BLOCK: u32 = 8;

/// Which block to report, counting from the most contrasty: using the 5th
/// strongest instead of the maximum ignores a few hot pixels or JPEG
/// artefacts that would otherwise make a blank frame look "detailed".
const RANK_FROM_TOP: usize = 5;

/// Strongest local contrast in the image: the (max − min) luminance range of
/// each 8×8 block, taking the 5th highest. Pass the small (≤ 640 px) copy —
/// the value is about content, not resolution.
pub fn strongest_local_contrast(img: &DynamicImage) -> f32 {
    let luma = img.to_luma8();
    let (w, h) = (luma.width() / BLOCK, luma.height() / BLOCK);
    if w == 0 || h == 0 {
        return 0.0;
    }

    let mut ranges: Vec<u8> = Vec::with_capacity((w * h) as usize);
    for by in 0..h {
        for bx in 0..w {
            let (mut lo, mut hi) = (u8::MAX, u8::MIN);
            for y in by * BLOCK..(by + 1) * BLOCK {
                for x in bx * BLOCK..(bx + 1) * BLOCK {
                    let v = luma.get_pixel(x, y).0[0];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            ranges.push(hi - lo);
        }
    }

    ranges.sort_unstable();
    let index = ranges.len().saturating_sub(RANK_FROM_TOP);
    ranges[index] as f32
}

/// True if the frame has no usable content (black or blank).
pub fn is_unusable(img: &DynamicImage) -> bool {
    strongest_local_contrast(img) < UNUSABLE_LOCAL_CONTRAST
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    /// Deterministic low-level noise, like a real sensor on a flat surface.
    fn noisy(w: u32, h: u32, base: u8, amplitude: u8) -> GrayImage {
        let mut seed: u32 = 7;
        GrayImage::from_fn(w, h, |_, _| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            Luma([base.saturating_add((seed >> 24) as u8 % (amplitude + 1))])
        })
    }

    #[test]
    fn test_blank_grey_wall_is_unusable() {
        let img = DynamicImage::ImageLuma8(noisy(640, 427, 120, 8));
        assert!(is_unusable(&img));
    }

    #[test]
    fn test_black_frame_is_unusable() {
        let img = DynamicImage::ImageLuma8(noisy(640, 427, 5, 6));
        assert!(is_unusable(&img));
    }

    #[test]
    fn test_small_bird_in_clear_sky_is_usable() {
        // Bright, nearly uniform sky with one small dark bird (16×10 px at
        // 640 px wide — the size of the bird-speck shots in the test shoot).
        let mut img = noisy(640, 427, 180, 6);
        for y in 200..210 {
            for x in 300..316 {
                img.put_pixel(x, y, Luma([40]));
            }
        }
        assert!(!is_unusable(&DynamicImage::ImageLuma8(img)));
    }

    #[test]
    fn test_a_few_hot_pixels_do_not_count_as_content() {
        let mut img = noisy(640, 427, 120, 8);
        for i in 0..3 {
            img.put_pixel(50 + i * 100, 60, Luma([255]));
        }
        assert!(is_unusable(&DynamicImage::ImageLuma8(img)));
    }

    #[test]
    fn test_tiny_image_is_unusable() {
        assert_eq!(
            strongest_local_contrast(&DynamicImage::new_luma8(4, 4)),
            0.0
        );
    }
}
