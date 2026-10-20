//! Blink detection with MediaPipe FaceMesh V2 + Blendshape V2 (Apache-2.0).
//!
//! 1. FaceMesh V2 finds 478 landmarks on a square, levelled 256×256 face crop
//!    and reports how confident it is that the crop shows a usable face.
//! 2. Blendshape V2 turns 146 of those landmarks into 52 expression scores,
//!    including `eyeBlinkLeft` / `eyeBlinkRight` (0 = open, 1 = closed).
//!
//! This replaced a FaceMesh-v1 Eye-Aspect-Ratio heuristic. On hand-verified
//! faces from a real concert shoot (10 closed, 33 open) the blink scores
//! separate open from closed eyes with AUC 0.94 (EAR: 0.45, at best 0.84),
//! and the "clear blink" rule below caught 7/10 blinks with 0/33 open eyes
//! wrongly rejected (EAR: 4/10 caught, 3 wrong).

use anyhow::{anyhow, Context, Result};
use keeper_core::types::BoundingBox;
use ndarray::{Array3, Array4};
use std::path::Path;
use tracing::{debug, info};

use crate::preprocess;
use crate::runtime::OnnxModel;

/// FaceMesh V2 model file inside the models directory.
pub const LANDMARK_MODEL_FILE: &str = "face_landmarks_detector.onnx";
/// Blendshape V2 model file inside the models directory.
pub const BLENDSHAPE_MODEL_FILE: &str = "face_blendshapes.onnx";

/// FaceMesh V2 input: 256×256 RGB, values in [0, 1] (its native range — the
/// model's face confidence is clearly higher than with [-1, 1]).
const INPUT_SIZE: u32 = 256;

/// Crop side = this × the longer side of the face box, i.e. the 25 % margin
/// on each side that the FaceMesh V2 model card asks for.
const FACE_CROP_SCALE: f32 = 1.5;

/// Below this face confidence the landmarks — and so the blink scores — are
/// unreliable (extreme tilt, profile, occlusion). Such faces are never
/// rejected as blinks.
pub const MIN_FACE_CONFIDENCE: f32 = 0.5;

const NUM_LANDMARKS: usize = 478;

/// Positions of `eyeBlinkLeft` / `eyeBlinkRight` in the 52 blendshape outputs
/// (index 0 is `_neutral`). Verified on labelled closed eyes.
const EYE_BLINK_LEFT: usize = 9;
const EYE_BLINK_RIGHT: usize = 10;

/// The 146 FaceMesh landmarks the Blendshape V2 model takes as input, in
/// order (from the Blendshape V2 model card appendix).
#[rustfmt::skip]
const BLENDSHAPE_LANDMARKS: [usize; 146] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157,
    158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282,
    283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356,
    361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405,
    409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
];

#[derive(Debug, Clone)]
pub struct BlinkResult {
    /// True only for a *clear* blink — see [`is_clear_blink`].
    pub blink_detected: bool,
    /// `eyeBlinkLeft` / `eyeBlinkRight`: 0 = open, 1 = closed.
    pub left_blink: f32,
    pub right_blink: f32,
    /// FaceMesh's confidence (0..1) that the crop shows a usable face.
    pub face_confidence: f32,
    /// The less-closed eye's score — a blink means *both* eyes are closed.
    pub blink_score: f32,
}

pub struct BlinkDetector {
    landmarks: OnnxModel,
    blendshapes: OnnxModel,
}

/// A blink is rejected only when FaceMesh is confident it sees a usable face
/// and **both** eyes score as closed at least `threshold`. Precision matters
/// more than recall: wrongly rejecting a good shot is worse than missing a
/// blink.
pub fn is_clear_blink(face_confidence: f32, left: f32, right: f32, threshold: f64) -> bool {
    face_confidence >= MIN_FACE_CONFIDENCE && left.min(right) as f64 >= threshold
}

impl BlinkDetector {
    /// Loads both models from `models_dir`.
    pub fn new(models_dir: &Path) -> Result<Self> {
        Self::with_threads(models_dir, crate::runtime::available_cores())
    }

    /// Same as [`new`](Self::new) with each ONNX session limited to
    /// `threads` threads (for running several detectors in parallel).
    pub fn with_threads(models_dir: &Path, threads: usize) -> Result<Self> {
        let landmarks = OnnxModel::load_with_threads(
            &models_dir.join(LANDMARK_MODEL_FILE),
            "FaceMesh V2",
            threads,
        )?;
        let blendshapes = OnnxModel::load_with_threads(
            &models_dir.join(BLENDSHAPE_MODEL_FILE),
            "Blendshape V2",
            threads,
        )?;
        info!("Blink detector ready (FaceMesh V2 + Blendshape V2).");
        Ok(BlinkDetector {
            landmarks,
            blendshapes,
        })
    }

    pub fn detect_blink(
        &mut self,
        jpeg_bytes: &[u8],
        face_bbox: &BoundingBox,
        threshold: f64,
    ) -> Result<BlinkResult> {
        let img =
            preprocess::decode_jpeg(jpeg_bytes).context("Blink detector: failed to decode JPEG")?;
        self.detect_blink_image(&img, face_bbox, 0.0, threshold)
    }

    /// Same as [`detect_blink`](Self::detect_blink) on an already-decoded
    /// image. `face_bbox` must be in `img`'s pixel coordinates; `roll` is the
    /// head's tilt in radians (from the face detector's eye keypoints; 0 if
    /// unknown) — the model needs the eyes level.
    pub fn detect_blink_image(
        &mut self,
        img: &image::DynamicImage,
        face_bbox: &BoundingBox,
        roll: f32,
        threshold: f64,
    ) -> Result<BlinkResult> {
        // ---- Pass 1: crop from the face box, levelled by detector keypoints ---
        let coarse = CropParams {
            cx: face_bbox.x + face_bbox.width / 2.0,
            cy: face_bbox.y + face_bbox.height / 2.0,
            side: face_bbox.width.max(face_bbox.height) * FACE_CROP_SCALE,
            roll,
        };
        let (coarse_landmarks, _) = self.run_facemesh(img, &coarse)?;

        // ---- Pass 2: MediaPipe's tracking step -------------------------------
        // Re-centre on the landmarks and re-level on the iris centres, then run
        // again. Measured on the labelled faces this second pass is what gets
        // blink scores to their best separation.
        let refined = refine_crop(&coarse_landmarks, &coarse);
        let (landmarks, face_confidence) = self.run_facemesh(img, &refined)?;

        // The blendshape model takes (x, y) of its 146-landmark subset.
        let mut points = Array3::<f32>::zeros((1, BLENDSHAPE_LANDMARKS.len(), 2));
        for (i, &idx) in BLENDSHAPE_LANDMARKS.iter().enumerate() {
            points[[0, i, 0]] = landmarks[idx * 3];
            points[[0, i, 1]] = landmarks[idx * 3 + 1];
        }

        // ---- Blendshape V2: eyeBlinkLeft / eyeBlinkRight ---------------------
        let points = ort::value::Tensor::from_array(points)
            .map_err(|e| anyhow!("Failed to create blendshape input tensor: {e}"))?;
        let outputs = self
            .blendshapes
            .session
            .run(ort::inputs![points])
            .map_err(|e| anyhow!("Blendshape V2 inference failed: {e}"))?;
        let (_, scores) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to read blendshape scores: {e}"))?;
        if scores.len() <= EYE_BLINK_RIGHT {
            return Err(anyhow!(
                "Blendshape V2 returned only {} scores",
                scores.len()
            ));
        }
        let (left_blink, right_blink) = (scores[EYE_BLINK_LEFT], scores[EYE_BLINK_RIGHT]);

        let blink_detected = is_clear_blink(face_confidence, left_blink, right_blink, threshold);
        debug!(
            "Blink: left={:.2} right={:.2} face_conf={:.2} → blink={}",
            left_blink, right_blink, face_confidence, blink_detected
        );

        Ok(BlinkResult {
            blink_detected,
            left_blink,
            right_blink,
            face_confidence,
            blink_score: left_blink.min(right_blink),
        })
    }

    /// Runs FaceMesh V2 on one crop. Returns the 478 (x, y, z) landmarks in
    /// crop pixels, flattened, and the face confidence (0..1).
    fn run_facemesh(
        &mut self,
        img: &image::DynamicImage,
        crop: &CropParams,
    ) -> Result<(Vec<f32>, f32)> {
        let patch =
            preprocess::sample_square(img, crop.cx, crop.cy, crop.side, crop.roll, INPUT_SIZE);

        let n = INPUT_SIZE as usize;
        let mut input = Array4::<f32>::zeros((1, n, n, 3));
        for (x, y, pixel) in patch.enumerate_pixels() {
            for c in 0..3 {
                input[[0, y as usize, x as usize, c]] = pixel.0[c] as f32 / 255.0;
            }
        }
        let input = ort::value::Tensor::from_array(input)
            .map_err(|e| anyhow!("Failed to create FaceMesh input tensor: {e}"))?;
        let outputs = self
            .landmarks
            .session
            .run(ort::inputs![input])
            .map_err(|e| anyhow!("FaceMesh V2 inference failed: {e}"))?;

        let (_, lm) = outputs["Identity"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to read FaceMesh landmarks: {e}"))?;
        if lm.len() < NUM_LANDMARKS * 3 {
            return Err(anyhow!(
                "FaceMesh V2 returned {} values, expected {}",
                lm.len(),
                NUM_LANDMARKS * 3
            ));
        }
        let (_, flag) = outputs["Identity_1"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to read FaceMesh face flag: {e}"))?;
        // The face flag is a logit; convert to a probability.
        let confidence = 1.0 / (1.0 + (-flag[0]).exp());

        Ok((lm[..NUM_LANDMARKS * 3].to_vec(), confidence))
    }
}

/// A square crop of the source image: centre, side length (source pixels)
/// and rotation (radians, clockwise in image coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
struct CropParams {
    cx: f32,
    cy: f32,
    side: f32,
    roll: f32,
}

/// Iris-centre landmarks in FaceMesh V2 (the eye on the image's left, then
/// the eye on the image's right).
const IRIS_CENTRE_LEFT_IN_IMAGE: usize = 468;
const IRIS_CENTRE_RIGHT_IN_IMAGE: usize = 473;

/// Maps a point in crop pixels back to source-image pixels.
fn crop_to_source(crop: &CropParams, u: f32, v: f32) -> (f32, f32) {
    let scale = crop.side / INPUT_SIZE as f32;
    let half = INPUT_SIZE as f32 / 2.0;
    let (dx, dy) = ((u - half) * scale, (v - half) * scale);
    let (sin, cos) = crop.roll.sin_cos();
    (crop.cx + dx * cos - dy * sin, crop.cy + dx * sin + dy * cos)
}

/// Second-pass crop from first-pass landmarks: centred on the landmarks'
/// mean and rotated so the iris centres are level. Same side length.
fn refine_crop(landmarks: &[f32], crop: &CropParams) -> CropParams {
    let n = (landmarks.len() / 3) as f32;
    let (sum_u, sum_v) = landmarks
        .as_chunks::<3>()
        .0
        .iter()
        .fold((0.0, 0.0), |(su, sv), p| (su + p[0], sv + p[1]));
    let (cx, cy) = crop_to_source(crop, sum_u / n, sum_v / n);

    let iris = |i: usize| (landmarks[i * 3], landmarks[i * 3 + 1]);
    let (lu, lv) = iris(IRIS_CENTRE_LEFT_IN_IMAGE);
    let (ru, rv) = iris(IRIS_CENTRE_RIGHT_IN_IMAGE);
    // The crop is already rotated by `crop.roll`; add the remaining tilt.
    let roll = crop.roll + (rv - lv).atan2(ru - lu);

    CropParams {
        cx,
        cy,
        side: crop.side,
        roll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blendshape_landmark_subset_is_valid() {
        assert_eq!(BLENDSHAPE_LANDMARKS.len(), 146);
        assert!(
            BLENDSHAPE_LANDMARKS.windows(2).all(|w| w[0] < w[1]),
            "sorted, no duplicates"
        );
        assert!(*BLENDSHAPE_LANDMARKS.last().unwrap() < NUM_LANDMARKS);
    }

    #[test]
    fn test_clear_blink_needs_both_eyes_closed() {
        assert!(is_clear_blink(0.9, 0.8, 0.7, 0.4));
        // One eye open (a wink, or a foreshortened eye) is not a clear blink.
        assert!(!is_clear_blink(0.9, 0.8, 0.2, 0.4));
        assert!(!is_clear_blink(0.9, 0.2, 0.8, 0.4));
    }

    #[test]
    fn test_unsure_face_is_never_a_blink() {
        // Extreme tilt / profile: FaceMesh isn't confident → don't reject.
        assert!(!is_clear_blink(0.45, 0.9, 0.9, 0.4));
        assert!(is_clear_blink(0.5, 0.9, 0.9, 0.4));
    }

    /// 478 landmarks all at (u, v), with the two iris centres overridden.
    fn landmarks_at(u: f32, v: f32, left_iris: (f32, f32), right_iris: (f32, f32)) -> Vec<f32> {
        let mut lm: Vec<f32> = (0..NUM_LANDMARKS).flat_map(|_| [u, v, 0.0]).collect();
        lm[IRIS_CENTRE_LEFT_IN_IMAGE * 3] = left_iris.0;
        lm[IRIS_CENTRE_LEFT_IN_IMAGE * 3 + 1] = left_iris.1;
        lm[IRIS_CENTRE_RIGHT_IN_IMAGE * 3] = right_iris.0;
        lm[IRIS_CENTRE_RIGHT_IN_IMAGE * 3 + 1] = right_iris.1;
        lm
    }

    #[test]
    fn test_refine_crop_keeps_a_centred_level_face() {
        let crop = CropParams {
            cx: 500.0,
            cy: 300.0,
            side: 512.0,
            roll: 0.0,
        };
        let lm = landmarks_at(128.0, 128.0, (100.0, 110.0), (156.0, 110.0));
        let refined = refine_crop(&lm, &crop);
        // Mean is pulled very slightly by the two iris points; roll stays 0.
        assert!((refined.cx - 500.0).abs() < 0.2 && (refined.cy - 300.0).abs() < 0.2);
        assert!(refined.roll.abs() < 1e-6);
        assert_eq!(refined.side, 512.0);
    }

    #[test]
    fn test_refine_crop_recentres_and_relevels() {
        // Side 512 over a 256 crop → 2 source px per crop px.
        let crop = CropParams {
            cx: 500.0,
            cy: 300.0,
            side: 512.0,
            roll: 0.0,
        };
        // Face 10 crop px right of centre; eyes tilted 45° (right eye lower).
        let lm = landmarks_at(138.0, 128.0, (120.0, 120.0), (140.0, 140.0));
        let refined = refine_crop(&lm, &crop);
        assert!((refined.cx - 520.0).abs() < 0.5, "cx {}", refined.cx);
        assert!((refined.roll - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    }

    #[test]
    fn test_crop_to_source_respects_rotation() {
        let crop = CropParams {
            cx: 100.0,
            cy: 100.0,
            side: 256.0,
            roll: std::f32::consts::FRAC_PI_2,
        };
        // 10 px right of centre in a crop rotated 90° clockwise = 10 px down.
        let (x, y) = crop_to_source(&crop, 138.0, 128.0);
        assert!((x - 100.0).abs() < 1e-4 && (y - 110.0).abs() < 1e-4);
    }

    #[test]
    fn test_threshold_is_inclusive() {
        assert!(is_clear_blink(1.0, 0.4, 0.4, 0.4));
        assert!(!is_clear_blink(1.0, 0.39, 0.9, 0.4));
    }
}
