use anyhow::{Context, Result};
use keeper_core::types::BoundingBox;
use ndarray::Array4;
use ort::session::SessionOutputs;
use tracing::{debug, info, warn};

use crate::preprocess;
use crate::runtime::OnnxModel;

#[derive(Debug, Clone)]
pub struct BlinkResult {
    pub blink_detected: bool,
    pub right_ear: f64,
    pub left_ear: f64,
    pub average_ear: f64,
}

pub struct BlinkDetector {
    model: OnnxModel,
    is_nchw: bool,
    input_size: u32,
}

const RIGHT_EYE_LANDMARKS: [usize; 6] = [33, 159, 158, 133, 153, 145];
const LEFT_EYE_LANDMARKS: [usize; 6] = [362, 380, 374, 263, 386, 385];

impl BlinkDetector {
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let model = OnnxModel::load(model_path, "FaceMesh")?;
        let is_nchw = false;

        info!(
            "Blink detector loaded (input format: {})",
            if is_nchw { "NCHW" } else { "NHWC" }
        );

        Ok(BlinkDetector {
            model,
            is_nchw,
            input_size: 192,
        })
    }

    pub fn detect_blink(
        &mut self,
        jpeg_bytes: &[u8],
        face_bbox: &BoundingBox,
        ear_threshold: f64,
    ) -> Result<BlinkResult> {
        let img =
            preprocess::decode_jpeg(jpeg_bytes).context("Blink detector: failed to decode JPEG")?;

        let face_crop = self.crop_face_with_margin(&img, face_bbox);

        let resized = preprocess::resize_image(&face_crop, self.input_size, self.input_size);

        let tensor = if self.is_nchw {
            preprocess::image_to_nchw_tensor(
                &resized,
                preprocess::SIMPLE_MEAN,
                preprocess::SIMPLE_STD,
            )
        } else {
            self.image_to_nhwc_tensor(&resized)
        };

        debug!("FaceMesh input tensor shape: {:?}", tensor.shape());

        let input_value = ort::value::Tensor::from_array(tensor)
            .context("Failed to create FaceMesh input tensor")?;
        let outputs = self
            .model
            .session
            .run(ort::inputs![input_value])
            .context("FaceMesh inference failed")?;

        let landmarks = extract_landmarks(&outputs)?;

        if landmarks.len() < 468 {
            warn!(
                "Expected 468 landmarks, got {}. Model output may be wrong.",
                landmarks.len()
            );
            return Ok(BlinkResult {
                blink_detected: false,
                right_ear: 1.0,
                left_ear: 1.0,
                average_ear: 1.0,
            });
        }

        let right_ear = compute_ear(&landmarks, &RIGHT_EYE_LANDMARKS);
        let left_ear = compute_ear(&landmarks, &LEFT_EYE_LANDMARKS);
        let average_ear = (right_ear + left_ear) / 2.0;

        let blink_detected = right_ear < ear_threshold && left_ear < ear_threshold;

        debug!(
            "EAR: right={:.3}, left={:.3}, avg={:.3}, blink={}",
            right_ear, left_ear, average_ear, blink_detected
        );

        Ok(BlinkResult {
            blink_detected,
            right_ear,
            left_ear,
            average_ear,
        })
    }

    fn crop_face_with_margin(
        &self,
        img: &image::DynamicImage,
        bbox: &BoundingBox,
    ) -> image::DynamicImage {
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);

        let margin_x = bbox.width * 0.25;
        let margin_y = bbox.height * 0.25;

        let x1 = (bbox.x - margin_x).max(0.0) as u32;
        let y1 = (bbox.y - margin_y).max(0.0) as u32;
        let x2 = (bbox.x + bbox.width + margin_x).min(img_w) as u32;
        let y2 = (bbox.y + bbox.height + margin_y).min(img_h) as u32;

        let crop_w = x2.saturating_sub(x1).max(1);
        let crop_h = y2.saturating_sub(y1).max(1);

        img.crop_imm(x1, y1, crop_w, crop_h)
    }

    fn image_to_nhwc_tensor(&self, img: &image::DynamicImage) -> Array4<f32> {
        let rgb = img.to_rgb8();
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);

        let mut tensor = Array4::<f32>::zeros((1, height, width, 3));

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, y, x, c]] = pixel.0[c] as f32 / 255.0;
                }
            }
        }

        tensor
    }
}

fn extract_landmarks(outputs: &SessionOutputs<'_>) -> Result<Vec<[f64; 2]>> {
    let output_tensor = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract FaceMesh output")?;

    let (shape, data): (&ort::value::Shape, &[f32]) = output_tensor;
    debug!(
        "FaceMesh output shape: {:?}, data length: {}",
        shape,
        data.len()
    );

    let mut landmarks: Vec<[f64; 2]> = Vec::new();

    if data.len() >= 1404 {
        for i in 0..468 {
            let x = data[i * 3] as f64;
            let y = data[i * 3 + 1] as f64;
            landmarks.push([x, y]);
        }
    } else if data.len() >= 936 {
        for i in 0..468 {
            let x = data[i * 2] as f64;
            let y = data[i * 2 + 1] as f64;
            landmarks.push([x, y]);
        }
    } else {
        warn!(
            "Unexpected FaceMesh output size: {} (expected >= 1404)",
            data.len()
        );
    }

    Ok(landmarks)
}

fn compute_ear(landmarks: &[[f64; 2]], indices: &[usize; 6]) -> f64 {
    for &idx in indices {
        if idx >= landmarks.len() {
            return 1.0;
        }
    }

    let p1 = landmarks[indices[0]];
    let p2 = landmarks[indices[1]];
    let p3 = landmarks[indices[2]];
    let p4 = landmarks[indices[3]];
    let p5 = landmarks[indices[4]];
    let p6 = landmarks[indices[5]];

    let vertical_a = euclidean_distance(p2, p6);
    let vertical_b = euclidean_distance(p3, p5);
    let horizontal = euclidean_distance(p1, p4);

    if horizontal < 1e-6 {
        return 1.0;
    }

    (vertical_a + vertical_b) / (2.0 * horizontal)
}

fn euclidean_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ear_open_eye() {
        let mut landmarks = vec![[0.0, 0.0]; 468];

        landmarks[33] = [0.3, 0.4];
        landmarks[159] = [0.35, 0.35];
        landmarks[158] = [0.4, 0.35];
        landmarks[133] = [0.45, 0.4];
        landmarks[153] = [0.4, 0.45];
        landmarks[145] = [0.35, 0.45];

        let ear = compute_ear(&landmarks, &RIGHT_EYE_LANDMARKS);
        assert!(ear > 0.21, "Open eye EAR ({:.3}) should be > 0.21", ear);
    }

    #[test]
    fn test_ear_closed_eye() {
        let mut landmarks = vec![[0.0, 0.0]; 468];

        landmarks[33] = [0.3, 0.4];
        landmarks[159] = [0.35, 0.40];
        landmarks[158] = [0.4, 0.40];
        landmarks[133] = [0.45, 0.4];
        landmarks[153] = [0.4, 0.41];
        landmarks[145] = [0.35, 0.41];

        let ear = compute_ear(&landmarks, &RIGHT_EYE_LANDMARKS);
        assert!(ear < 0.21, "Closed eye EAR ({:.3}) should be < 0.21", ear);
    }

    #[test]
    fn test_ear_symmetry() {
        let mut landmarks = vec![[0.0, 0.0]; 468];

        landmarks[33] = [0.3, 0.4];
        landmarks[159] = [0.35, 0.35];
        landmarks[158] = [0.4, 0.35];
        landmarks[133] = [0.45, 0.4];
        landmarks[153] = [0.4, 0.45];
        landmarks[145] = [0.35, 0.45];

        landmarks[362] = [0.55, 0.4];
        landmarks[380] = [0.6, 0.35];
        landmarks[374] = [0.65, 0.35];
        landmarks[263] = [0.7, 0.4];
        landmarks[386] = [0.65, 0.45];
        landmarks[385] = [0.6, 0.45];

        let right_ear = compute_ear(&landmarks, &RIGHT_EYE_LANDMARKS);
        let left_ear = compute_ear(&landmarks, &LEFT_EYE_LANDMARKS);

        let diff = (right_ear - left_ear).abs();
        assert!(
            diff < 0.01,
            "Symmetric eyes should have similar EAR (diff={:.3})",
            diff
        );
    }

    #[test]
    fn test_blink_requires_both_eyes() {
        let threshold = 0.21;

        let right_ear = 0.15;
        let left_ear = 0.30;
        let blink = right_ear < threshold && left_ear < threshold;
        assert!(
            !blink,
            "Wink (one eye closed) should not be flagged as blink"
        );

        let right_ear = 0.15;
        let left_ear = 0.12;
        let blink = right_ear < threshold && left_ear < threshold;
        assert!(blink, "Both eyes closed should be flagged as blink");
    }
}
