//! "Accidental shot" detection with CLIP ViT-B/32 (OpenAI, MIT licence).
//!
//! Test shots and photos taken by mistake (the floor, shoes, a desk, a phone
//! on a bed) are sharp and well exposed, so no pixel measurement can tell
//! them from real photos. CLIP can: it turns a photo into an "embedding" (512
//! numbers describing what's in it) that can be compared with embeddings of
//! short descriptions such as "a test shot of random objects at home" or
//! "a concert photo". The share of probability that lands on the accidental
//! descriptions is the photo's **accidental score** (0..1).
//!
//! The descriptions are encoded once, offline, by `models/convert_clip.py`
//! into `clip_prompts.json`, so the app only needs CLIP's image half.
//!
//! Measured in the full pipeline on a real 118-photo test folder (photos with
//! a face skipped, see `pipeline.rs`): 8 of 9 accidental shots scored
//! 0.57–0.97 (the 9th shows faces on a screen, so it is skipped); the highest
//! real photo scored 0.34. Precision matters most: a flagged photo is only kept from
//! becoming the keeper (it stays unrated) — it is never rejected.

use anyhow::{anyhow, Context, Result};
use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::Array4;
use serde::Deserialize;
use std::path::Path;
use tracing::info;

use crate::runtime::OnnxModel;

/// CLIP image encoder (int8 weights) inside the models directory.
pub const IMAGE_MODEL_FILE: &str = "clip_vit_b32_image.onnx";
/// Pre-encoded descriptions inside the models directory.
pub const PROMPTS_FILE: &str = "clip_prompts.json";

/// Photos scoring at least this are treated as likely accidental. Above the
/// highest real photo measured (0.34), below the lowest accidental one (0.57).
pub const ACCIDENTAL_THRESHOLD: f64 = 0.5;

/// CLIP's input: 224×224 RGB, normalised with these per-channel values.
const INPUT_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Length of a CLIP ViT-B/32 embedding.
const EMBEDDING_LEN: usize = 512;

#[derive(Deserialize)]
struct PromptFile {
    logit_scale: f32,
    accidental: Vec<Prompt>,
    deliberate: Vec<Prompt>,
}

#[derive(Deserialize)]
struct Prompt {
    embedding: Vec<f32>,
}

pub struct AccidentalDetector {
    model: OnnxModel,
    accidental: Vec<Vec<f32>>,
    deliberate: Vec<Vec<f32>>,
    logit_scale: f32,
}

impl AccidentalDetector {
    /// Loads the CLIP image encoder and the pre-encoded descriptions from
    /// `models_dir`, with the ONNX session limited to `threads` threads.
    pub fn with_threads(models_dir: &Path, threads: usize) -> Result<Self> {
        let model = OnnxModel::load_with_threads(
            &models_dir.join(IMAGE_MODEL_FILE),
            "CLIP ViT-B/32 image encoder",
            threads,
        )?;

        let path = models_dir.join(PROMPTS_FILE);
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("Failed to read {:?}", path))?;
        let prompts: PromptFile =
            serde_json::from_str(&text).with_context(|| format!("Failed to parse {:?}", path))?;

        let accidental: Vec<Vec<f32>> = prompts
            .accidental
            .into_iter()
            .map(|p| p.embedding)
            .collect();
        let deliberate: Vec<Vec<f32>> = prompts
            .deliberate
            .into_iter()
            .map(|p| p.embedding)
            .collect();
        if accidental.is_empty() || deliberate.is_empty() {
            return Err(anyhow!(
                "{:?} must list accidental and deliberate prompts",
                path
            ));
        }
        if accidental
            .iter()
            .chain(&deliberate)
            .any(|e| e.len() != EMBEDDING_LEN)
        {
            return Err(anyhow!(
                "{:?}: every embedding must have {EMBEDDING_LEN} values",
                path
            ));
        }

        info!(
            "Accidental-shot detector ready ({} accidental / {} deliberate descriptions).",
            accidental.len(),
            deliberate.len()
        );
        Ok(AccidentalDetector {
            model,
            accidental,
            deliberate,
            logit_scale: prompts.logit_scale,
        })
    }

    /// The photo's accidental score (0..1). Pass the small (≤ 640 px) copy —
    /// CLIP only looks at 224×224.
    pub fn score(&mut self, img: &DynamicImage) -> Result<f32> {
        let input = ort::value::Tensor::from_array(clip_input(img))
            .map_err(|e| anyhow!("Failed to create CLIP input tensor: {e}"))?;
        let outputs = self
            .model
            .session
            .run(ort::inputs![input])
            .map_err(|e| anyhow!("CLIP inference failed: {e}"))?;
        let (_, embedding) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to read CLIP embedding: {e}"))?;
        if embedding.len() != EMBEDDING_LEN {
            return Err(anyhow!(
                "CLIP returned {} values, expected {EMBEDDING_LEN}",
                embedding.len()
            ));
        }
        Ok(accidental_probability(
            embedding,
            &self.accidental,
            &self.deliberate,
            self.logit_scale,
        ))
    }
}

/// Share of probability on the accidental descriptions: cosine similarity
/// with every description, scaled by CLIP's temperature, then softmax over
/// all of them (accidental + deliberate) — CLIP's standard "zero-shot"
/// classification. Description embeddings must already be unit length.
pub fn accidental_probability(
    image_embedding: &[f32],
    accidental: &[Vec<f32>],
    deliberate: &[Vec<f32>],
    logit_scale: f32,
) -> f32 {
    let norm = image_embedding
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    let logit = |text: &Vec<f32>| {
        logit_scale
            * image_embedding
                .iter()
                .zip(text)
                .map(|(a, b)| a * b)
                .sum::<f32>()
            / norm
    };
    let acc: Vec<f32> = accidental.iter().map(logit).collect();
    let del: Vec<f32> = deliberate.iter().map(logit).collect();

    // Softmax, shifted by the max for numerical stability.
    let max = acc
        .iter()
        .chain(&del)
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let acc_sum: f32 = acc.iter().map(|v| (v - max).exp()).sum();
    let del_sum: f32 = del.iter().map(|v| (v - max).exp()).sum();
    acc_sum / (acc_sum + del_sum)
}

/// CLIP's preprocessing: scale the shorter side to 224 px (bicubic), take the
/// centre 224×224, normalise each channel. Returns a 1×3×224×224 tensor.
fn clip_input(img: &DynamicImage) -> Array4<f32> {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let scale = INPUT_SIZE as f32 / w.min(h) as f32;
    let (rw, rh) = (
        ((w as f32 * scale).round() as u32).max(INPUT_SIZE),
        ((h as f32 * scale).round() as u32).max(INPUT_SIZE),
    );
    let resized = img.resize_exact(rw, rh, FilterType::CatmullRom).to_rgb8();
    let (left, top) = ((rw - INPUT_SIZE) / 2, (rh - INPUT_SIZE) / 2);

    let n = INPUT_SIZE as usize;
    let mut input = Array4::<f32>::zeros((1, 3, n, n));
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let pixel = resized.get_pixel(left + x, top + y).0;
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] =
                    (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn test_photo_matching_accidental_description_scores_high() {
        let accidental = vec![unit(&[1.0, 0.0, 0.0])];
        let deliberate = vec![unit(&[0.0, 1.0, 0.0]), unit(&[0.0, 0.0, 1.0])];
        let p = accidental_probability(&[0.9, 0.1, 0.0], &accidental, &deliberate, 100.0);
        assert!(p > 0.99, "p = {p}");
    }

    #[test]
    fn test_photo_matching_deliberate_description_scores_low() {
        let accidental = vec![unit(&[1.0, 0.0, 0.0])];
        let deliberate = vec![unit(&[0.0, 1.0, 0.0])];
        let p = accidental_probability(&[0.1, 0.9, 0.0], &accidental, &deliberate, 100.0);
        assert!(p < 0.01, "p = {p}");
    }

    #[test]
    fn test_score_ignores_embedding_length() {
        // CLIP embeddings aren't unit length; only the direction matters.
        let accidental = vec![unit(&[1.0, 1.0])];
        let deliberate = vec![unit(&[1.0, -1.0])];
        let a = accidental_probability(&[0.3, 0.1], &accidental, &deliberate, 100.0);
        let b = accidental_probability(&[30.0, 10.0], &accidental, &deliberate, 100.0);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn test_equal_match_splits_by_description_count() {
        // An embedding equally far from everything: probability is shared
        // equally per description (1 accidental of 4 → 0.25).
        let accidental = vec![unit(&[1.0, 0.0])];
        let deliberate = vec![unit(&[1.0, 0.0]); 3];
        let p = accidental_probability(&[1.0, 0.0], &accidental, &deliberate, 100.0);
        assert!((p - 0.25).abs() < 1e-6, "p = {p}");
    }

    #[test]
    fn test_clip_input_is_224_square_and_normalised() {
        // A wide mid-grey image: shape 1×3×224×224, every value = (0.5−mean)/std.
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            640,
            427,
            image::Rgb([128, 128, 128]),
        ));
        let t = clip_input(&img);
        assert_eq!(t.shape(), &[1, 3, 224, 224]);
        for c in 0..3 {
            let expected = (128.0 / 255.0 - MEAN[c]) / STD[c];
            assert!((t[[0, c, 100, 100]] - expected).abs() < 1e-4);
        }
    }
}
