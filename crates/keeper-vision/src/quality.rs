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

// =============================================================================
// Blur: is ANY part of the photo in focus?
//
// The focus score used to pick keepers measures the eye region (or the
// centre), which can't tell "the subject is small or off-centre" from
// "nothing is sharp". Here we ask whether any region of the frame is in
// focus: the Laplacian variance of 32×32 tiles on a ≤ 1500 px copy, averaged
// over the 5 sharpest tiles. Camera-shake and missed-focus shots have no
// sharp tile anywhere; a small in-focus subject (a bird in clear sky) does.
//
// Calibrated on 118 real NEFs (av's shoot): genuinely sharp photos scored
// 1,092–22,140 (lowest: small birds in clear sky); the one motion-blurred
// shot 259. The same sharp photos blurred by σ = 1 px at this size (~4 px at
// full 24 MP) scored 145–431 (one landscape 1,057); σ ≥ 2 px scored ≤ 91.
// Deliberately conservative: a photo whose background is sharp but whose
// subject is soft is NOT caught here.
// =============================================================================

/// Below this peak sharpness the whole photo is out of focus or shaken.
/// Measured with this implementation on the 118 NEFs: the motion-blurred shot
/// scored 363, the lowest sharp photo (small bird in clear sky) 1,151 — the
/// threshold leans towards keeping photos (a wrong reject is worse than a
/// missed blur).
pub const BLURRY_PEAK_SHARPNESS: f32 = 500.0;

/// Long edge (px) the photo is scaled to before measuring, so the threshold
/// means the same for every camera resolution.
const SHARPNESS_LONG_EDGE: u32 = 1500;
const SHARPNESS_TILE: usize = 32;
const SHARPEST_TILES: usize = 5;

/// How sharp the sharpest part of the photo is: mean Laplacian variance of
/// the 5 sharpest 32×32 tiles, measured on a copy scaled to ≤ 1500 px.
pub fn peak_sharpness(img: &DynamicImage) -> f32 {
    let scaled;
    let img = if img.width().max(img.height()) > SHARPNESS_LONG_EDGE {
        scaled = img.thumbnail(SHARPNESS_LONG_EDGE, SHARPNESS_LONG_EDGE);
        &scaled
    } else {
        img
    };
    let gray = img.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w < SHARPNESS_TILE + 2 || h < SHARPNESS_TILE + 2 {
        return 0.0;
    }
    let px = gray.as_raw();

    // Laplacian of the interior pixels; tile (tx, ty) covers Laplacian
    // positions [tx·32, tx·32 + 32) × [ty·32, ty·32 + 32).
    let (lw, lh) = (w - 2, h - 2);
    let (tiles_x, tiles_y) = (lw / SHARPNESS_TILE, lh / SHARPNESS_TILE);
    let mut variances: Vec<f64> = Vec::with_capacity(tiles_x * tiles_y);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
            for ly in ty * SHARPNESS_TILE..(ty + 1) * SHARPNESS_TILE {
                for lx in tx * SHARPNESS_TILE..(tx + 1) * SHARPNESS_TILE {
                    let (x, y) = (lx + 1, ly + 1);
                    let lap = px[(y - 1) * w + x] as f64
                        + px[(y + 1) * w + x] as f64
                        + px[y * w + x - 1] as f64
                        + px[y * w + x + 1] as f64
                        - 4.0 * px[y * w + x] as f64;
                    sum += lap;
                    sum_sq += lap * lap;
                }
            }
            let n = (SHARPNESS_TILE * SHARPNESS_TILE) as f64;
            let mean = sum / n;
            variances.push(sum_sq / n - mean * mean);
        }
    }

    variances.sort_unstable_by(|a, b| b.total_cmp(a));
    let top = &variances[..variances.len().min(SHARPEST_TILES)];
    (top.iter().sum::<f64>() / top.len() as f64) as f32
}

/// True if no part of the photo is in focus (camera shake / missed focus).
pub fn is_blurry(img: &DynamicImage) -> bool {
    peak_sharpness(img) < BLURRY_PEAK_SHARPNESS
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

    /// Grey background with a sharp black/white checkerboard patch of
    /// `patch` px at the centre (a crisp, in-focus subject).
    fn sharp_subject(w: u32, h: u32, patch: u32) -> DynamicImage {
        let (x0, y0) = ((w - patch) / 2, (h - patch) / 2);
        DynamicImage::ImageLuma8(GrayImage::from_fn(w, h, |x, y| {
            let inside = x >= x0 && x < x0 + patch && y >= y0 && y < y0 + patch;
            if inside && ((x / 4) + (y / 4)) % 2 == 0 {
                Luma([20])
            } else if inside {
                Luma([235])
            } else {
                Luma([150])
            }
        }))
    }

    #[test]
    fn test_sharp_subject_is_not_blurry() {
        let img = sharp_subject(1500, 1000, 200);
        assert!(peak_sharpness(&img) > BLURRY_PEAK_SHARPNESS);
        assert!(!is_blurry(&img));
    }

    #[test]
    fn test_out_of_focus_photo_is_blurry() {
        let blurred = sharp_subject(1500, 1000, 200).blur(3.0);
        assert!(is_blurry(&blurred), "peak {}", peak_sharpness(&blurred));
    }

    #[test]
    fn test_tiny_sharp_subject_on_flat_sky_is_not_blurry() {
        // Like a small bird in a clear sky: most of the frame is featureless.
        let img = sharp_subject(1500, 1000, 48);
        assert!(!is_blurry(&img), "peak {}", peak_sharpness(&img));
    }

    #[test]
    fn test_large_preview_is_measured_on_a_downscaled_copy() {
        // A 6000 px preview of the same scene must not be judged blurrier
        // just because it has more pixels per detail.
        let big = sharp_subject(6000, 4000, 800).resize_exact(
            6000,
            4000,
            image::imageops::FilterType::Nearest,
        );
        assert!(!is_blurry(&big), "peak {}", peak_sharpness(&big));
    }

    #[test]
    fn test_tiny_image_is_unusable() {
        assert_eq!(
            strongest_local_contrast(&DynamicImage::new_luma8(4, 4)),
            0.0
        );
    }
}
