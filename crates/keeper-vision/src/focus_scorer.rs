use anyhow::{Context, Result};
use keeper_core::types::BoundingBox;
use tracing::{debug, warn};

use crate::preprocess;

#[derive(Debug, Clone)]
pub struct FocusResult {
    pub sharpness_score: f64,
    pub raw_variance: f64,
    pub noise_estimate: f64,
    pub scored_region: ScoredRegion,
}

#[derive(Debug, Clone)]
pub enum ScoredRegion {
    EyeRegion { face_bbox: BoundingBox },
    CenterFallback,
}

pub fn score_sharpness(jpeg_bytes: &[u8], face_bbox: Option<&BoundingBox>) -> Result<FocusResult> {
    let img = preprocess::decode_jpeg(jpeg_bytes).context("Focus scorer: failed to decode JPEG")?;

    let gray = img.to_luma8();

    match face_bbox {
        Some(bbox) => score_eye_region(&gray, bbox),
        None => score_center_region(&gray),
    }
}

fn score_eye_region(gray: &image::GrayImage, face_bbox: &BoundingBox) -> Result<FocusResult> {
    let (img_w, img_h) = (gray.width() as f32, gray.height() as f32);

    let x = face_bbox.x.max(0.0).min(img_w);
    let y = face_bbox.y.max(0.0).min(img_h);
    let w = face_bbox.width.min(img_w - x);
    let h = face_bbox.height.min(img_h - y);

    let eye_y = y;
    let eye_h = h * 0.4;

    let eye_x = x + w * 0.1;
    let eye_w = w * 0.8;

    let crop_x = eye_x as u32;
    let crop_y = eye_y as u32;
    let crop_w = (eye_w as u32).max(16);
    let crop_h = (eye_h as u32).max(16);

    let crop_w = crop_w.min(gray.width().saturating_sub(crop_x));
    let crop_h = crop_h.min(gray.height().saturating_sub(crop_y));

    if crop_w < 16 || crop_h < 16 {
        debug!(
            "Eye region too small ({}x{}), falling back to center",
            crop_w, crop_h
        );
        return score_center_region(gray);
    }

    let eye_crop = image::imageops::crop_imm(gray, crop_x, crop_y, crop_w, crop_h).to_image();

    let raw_variance = variance_of_laplacian(&eye_crop);
    let noise_estimate = estimate_noise(gray);
    let sharpness_score = if noise_estimate > 0.0 {
        raw_variance / noise_estimate
    } else {
        raw_variance
    };

    debug!(
        "Eye region score: {:.2} (raw={:.2}, noise={:.4}, crop={}x{})",
        sharpness_score, raw_variance, noise_estimate, crop_w, crop_h
    );

    Ok(FocusResult {
        sharpness_score,
        raw_variance,
        noise_estimate,
        scored_region: ScoredRegion::EyeRegion {
            face_bbox: *face_bbox,
        },
    })
}

fn score_center_region(gray: &image::GrayImage) -> Result<FocusResult> {
    let (w, h) = (gray.width(), gray.height());

    let crop_x = w / 4;
    let crop_y = h / 4;
    let crop_w = w / 2;
    let crop_h = h / 2;

    if crop_w < 16 || crop_h < 16 {
        warn!("Image too small for focus scoring ({}x{})", w, h);
        return Ok(FocusResult {
            sharpness_score: 0.0,
            raw_variance: 0.0,
            noise_estimate: 0.0,
            scored_region: ScoredRegion::CenterFallback,
        });
    }

    let center_crop = image::imageops::crop_imm(gray, crop_x, crop_y, crop_w, crop_h).to_image();

    let raw_variance = variance_of_laplacian(&center_crop);
    let noise_estimate = estimate_noise(gray);
    let sharpness_score = if noise_estimate > 0.0 {
        raw_variance / noise_estimate
    } else {
        raw_variance
    };

    debug!(
        "Center region score: {:.2} (raw={:.2}, noise={:.4})",
        sharpness_score, raw_variance, noise_estimate
    );

    Ok(FocusResult {
        sharpness_score,
        raw_variance,
        noise_estimate,
        scored_region: ScoredRegion::CenterFallback,
    })
}

fn variance_of_laplacian(gray: &image::GrayImage) -> f64 {
    let (w, h) = (gray.width() as usize, gray.height() as usize);

    if w < 3 || h < 3 {
        return 0.0;
    }

    let pixels = gray.as_raw();
    let mut laplacian_values: Vec<f64> = Vec::with_capacity((w - 2) * (h - 2));

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = pixels[y * w + x] as f64;
            let top = pixels[(y - 1) * w + x] as f64;
            let bottom = pixels[(y + 1) * w + x] as f64;
            let left = pixels[y * w + (x - 1)] as f64;
            let right = pixels[y * w + (x + 1)] as f64;

            let laplacian = top + bottom + left + right - 4.0 * center;
            laplacian_values.push(laplacian);
        }
    }

    if laplacian_values.is_empty() {
        return 0.0;
    }

    let n = laplacian_values.len() as f64;
    let sum: f64 = laplacian_values.iter().sum();
    let sum_sq: f64 = laplacian_values.iter().map(|v| v * v).sum();
    let mean = sum / n;
    sum_sq / n - mean * mean
}

fn estimate_noise(gray: &image::GrayImage) -> f64 {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let pixels = gray.as_raw();
    let patch_size = 16;

    if w < patch_size || h < patch_size {
        return 1.0;
    }

    let mut min_std = f64::MAX;
    let mut flattest_patch: Vec<f64> = Vec::new();

    for py in (0..h - patch_size).step_by(patch_size) {
        for px in (0..w - patch_size).step_by(patch_size) {
            let mut patch_vals: Vec<f64> = Vec::with_capacity(patch_size * patch_size);
            for y in py..py + patch_size {
                for x in px..px + patch_size {
                    patch_vals.push(pixels[y * w + x] as f64);
                }
            }

            let std_dev = compute_std_dev(&patch_vals);
            if std_dev < min_std {
                min_std = std_dev;
                flattest_patch = patch_vals;
            }
        }
    }

    if flattest_patch.is_empty() {
        return 1.0;
    }

    let median = compute_median(&flattest_patch);
    let abs_deviations: Vec<f64> = flattest_patch.iter().map(|v| (v - median).abs()).collect();
    let mad = compute_median(&abs_deviations);

    let noise_sigma = 1.4826 * mad;

    noise_sigma.max(0.1)
}

fn compute_std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

fn compute_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

pub fn pick_keeper(scores: &[(u64, FocusResult)]) -> Option<u64> {
    scores
        .iter()
        .max_by(|a, b| {
            a.1.sharpness_score
                .partial_cmp(&b.1.sharpness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(id, _)| *id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(width: u32, height: u32, pattern: &str) -> image::GrayImage {
        let mut img = image::GrayImage::new(width, height);

        match pattern {
            "sharp_edges" => {
                for y in 0..height {
                    for x in 0..width {
                        let val = if x % 4 < 2 { 0u8 } else { 255u8 };
                        img.put_pixel(x, y, image::Luma([val]));
                    }
                }
            }
            "blurry" => {
                for y in 0..height {
                    for x in 0..width {
                        let val = ((x as f32 / width as f32) * 255.0) as u8;
                        img.put_pixel(x, y, image::Luma([val]));
                    }
                }
            }
            "flat" => {
                for y in 0..height {
                    for x in 0..width {
                        img.put_pixel(x, y, image::Luma([128u8]));
                    }
                }
            }
            _ => {}
        }

        img
    }

    #[test]
    fn test_sharp_image_scores_higher_than_blurry() {
        let sharp = make_test_image(200, 200, "sharp_edges");
        let blurry = make_test_image(200, 200, "blurry");
        let flat = make_test_image(200, 200, "flat");

        let sharp_var = variance_of_laplacian(&sharp);
        let blurry_var = variance_of_laplacian(&blurry);
        let flat_var = variance_of_laplacian(&flat);

        assert!(
            sharp_var > blurry_var,
            "Sharp ({:.2}) should score higher than blurry ({:.2})",
            sharp_var,
            blurry_var
        );

        assert!(
            blurry_var > flat_var,
            "Blurry ({:.2}) should score higher than flat ({:.2})",
            blurry_var,
            flat_var
        );

        assert!(
            flat_var < 1.0,
            "Flat image variance should be ~0, got {:.2}",
            flat_var
        );
    }

    #[test]
    fn test_noise_estimation_on_flat_image() {
        let flat = make_test_image(200, 200, "flat");
        let noise = estimate_noise(&flat);

        assert!(
            noise < 2.0,
            "Flat image noise should be low, got {:.4}",
            noise
        );
    }

    #[test]
    fn test_pick_keeper_selects_sharpest() {
        let scores = vec![
            (
                1,
                FocusResult {
                    sharpness_score: 10.5,
                    raw_variance: 100.0,
                    noise_estimate: 9.5,
                    scored_region: ScoredRegion::CenterFallback,
                },
            ),
            (
                2,
                FocusResult {
                    sharpness_score: 25.3,
                    raw_variance: 200.0,
                    noise_estimate: 7.9,
                    scored_region: ScoredRegion::CenterFallback,
                },
            ),
            (
                3,
                FocusResult {
                    sharpness_score: 15.0,
                    raw_variance: 150.0,
                    noise_estimate: 10.0,
                    scored_region: ScoredRegion::CenterFallback,
                },
            ),
        ];

        let keeper = pick_keeper(&scores);
        assert_eq!(keeper, Some(2), "Image 2 has the highest score");
    }

    #[test]
    fn test_pick_keeper_empty_input() {
        let scores: Vec<(u64, FocusResult)> = vec![];
        let keeper = pick_keeper(&scores);
        assert_eq!(keeper, None);
    }
}
