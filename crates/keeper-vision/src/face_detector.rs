use anyhow::{anyhow, Context, Result};
use keeper_core::types::BoundingBox;
use ndarray::Array4;
use tracing::{debug, info, warn};

use crate::preprocess;
use crate::runtime::OnnxModel;

#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    /// Facial keypoints in the same coordinates as `bbox`. For
    /// `yolov8n-face.onnx` there are 5, in this order: the eye on the
    /// image's left, the eye on the image's right, nose tip, left and right
    /// mouth corner.
    pub keypoints: Vec<Keypoint>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    /// Visibility score from the model (0..1).
    pub confidence: f32,
}

pub struct FaceDetector {
    model: OnnxModel,
    input_size: u32,
}

impl FaceDetector {
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let model = OnnxModel::load(model_path, "YOLOv8-face")?;

        info!("Face detector ready.");

        Ok(FaceDetector {
            model,
            input_size: 640,
        })
    }

    pub fn detect(
        &mut self,
        jpeg_bytes: &[u8],
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<FaceDetection>> {
        let img =
            preprocess::decode_jpeg(jpeg_bytes).context("Face detector: failed to decode JPEG")?;
        self.detect_image(&img, confidence_threshold, iou_threshold)
    }

    /// Same as [`detect`](Self::detect) on an already-decoded image. Boxes are
    /// returned in `img`'s pixel coordinates. Passing a small (≤ 640 px) copy
    /// avoids an expensive full-resolution letterbox resize.
    pub fn detect_image(
        &mut self,
        img: &image::DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<FaceDetection>> {
        let (padded, transform) = preprocess::letterbox(img, self.input_size, self.input_size);

        let tensor: Array4<f32> = preprocess::image_to_yolo_tensor(&padded);

        debug!(
            "Input tensor shape: {:?}, original image: {}x{}",
            tensor.shape(),
            img.width(),
            img.height()
        );

        let input_value = ort::value::Tensor::from_array(tensor)
            .map_err(|e| anyhow!("Failed to create input tensor: {e}"))?;
        let outputs = self
            .model
            .session
            .run(ort::inputs![input_value])
            .map_err(|e| anyhow!("YOLO inference failed: {e}"))?;

        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract YOLO output tensor: {e}"))?;

        let shape: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();
        debug!("Raw output shape: {:?}", shape);

        let raw_detections = parse_yolo_output(&shape, output_data, confidence_threshold);
        debug!(
            "Detections after confidence filter: {}",
            raw_detections.len()
        );

        let nms_detections = non_maximum_suppression(raw_detections, iou_threshold);
        debug!("Detections after NMS: {}", nms_detections.len());

        let final_detections: Vec<FaceDetection> = nms_detections
            .into_iter()
            .map(|det| {
                let (x, y, w, h) = transform.to_original_coords(det.cx, det.cy, det.w, det.h);
                let keypoints = det
                    .keypoints
                    .iter()
                    .map(|kp| {
                        let (kx, ky) = transform.to_original_point(kp.x, kp.y);
                        Keypoint {
                            x: kx,
                            y: ky,
                            confidence: kp.confidence,
                        }
                    })
                    .collect();
                FaceDetection {
                    bbox: BoundingBox {
                        x,
                        y,
                        width: w,
                        height: h,
                    },
                    confidence: det.confidence,
                    keypoints,
                }
            })
            .collect();

        Ok(final_detections)
    }

    pub fn model_input_names(&self) -> Vec<String> {
        self.model.input_names()
    }

    pub fn model_output_names(&self) -> Vec<String> {
        self.model.output_names()
    }
}

#[derive(Debug, Clone)]
struct RawDetection {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    confidence: f32,
    /// In model-input (letterboxed 640×640) coordinates.
    keypoints: Vec<Keypoint>,
}

/// Parses a YOLOv8-face output tensor of shape `[1, channels, predictions]`,
/// stored channel-major. Channel layout (single class — "face"):
///
/// | channels | meaning                                                   |
/// |----------|-----------------------------------------------------------|
/// | 0..4     | box centre x, centre y, width, height (input pixels)      |
/// | 4        | **face confidence** (0..1)                                |
/// | 5..      | optional keypoints, 3 values each (x, y, visibility) —    |
/// |          | `yolov8n-face.onnx` has 5 (eyes, nose, mouth corners)     |
///
/// Only channel 4 is a confidence. Keypoint x/y are pixel coordinates (up to
/// ~900); treating them as class scores made every image "contain" thousands
/// of faces.
fn parse_yolo_output(
    shape: &[usize],
    data: &[f32],
    confidence_threshold: f32,
) -> Vec<RawDetection> {
    if shape.len() < 3 || shape[1] < 5 {
        warn!(
            "Unexpected YOLO output shape: {:?} (need [1, >=5, N])",
            shape
        );
        return vec![];
    }

    let num_values = shape[1];
    let num_preds = shape[2];

    if !(num_values - 5).is_multiple_of(3) {
        warn!(
            "YOLO output has {} values/prediction — expected 5 + 3×keypoints. \
             Using channel 4 as the face confidence.",
            num_values
        );
    }

    debug!(
        "YOLO output: {} values/det, {} predictions, {} keypoint(s)",
        num_values,
        num_preds,
        (num_values - 5) / 3
    );

    let num_keypoints = (num_values - 5) / 3;
    let mut detections = Vec::new();

    for i in 0..num_preds {
        let confidence = data[4 * num_preds + i];
        if confidence < confidence_threshold {
            continue;
        }

        // Keypoint k occupies channels 5+3k (x), 6+3k (y), 7+3k (visibility).
        let keypoints = (0..num_keypoints)
            .map(|k| Keypoint {
                x: data[(5 + 3 * k) * num_preds + i],
                y: data[(6 + 3 * k) * num_preds + i],
                confidence: data[(7 + 3 * k) * num_preds + i],
            })
            .collect();

        detections.push(RawDetection {
            cx: data[i],
            cy: data[num_preds + i],
            w: data[num_preds * 2 + i],
            h: data[num_preds * 3 + i],
            confidence,
            keypoints,
        });
    }

    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    detections
}

fn non_maximum_suppression(
    mut detections: Vec<RawDetection>,
    iou_threshold: f32,
) -> Vec<RawDetection> {
    if detections.is_empty() {
        return detections;
    }

    let mut keep: Vec<RawDetection> = Vec::new();

    while !detections.is_empty() {
        let best = detections.remove(0);

        detections.retain(|det| {
            let iou = compute_iou(&best, det);
            iou <= iou_threshold
        });

        keep.push(best);
    }

    keep
}

fn compute_iou(a: &RawDetection, b: &RawDetection) -> f32 {
    let a_x1 = a.cx - a.w / 2.0;
    let a_y1 = a.cy - a.h / 2.0;
    let a_x2 = a.cx + a.w / 2.0;
    let a_y2 = a.cy + a.h / 2.0;

    let b_x1 = b.cx - b.w / 2.0;
    let b_y1 = b.cy - b.h / 2.0;
    let b_x2 = b.cx + b.w / 2.0;
    let b_y2 = b.cy + b.h / 2.0;

    let inter_x1 = a_x1.max(b_x1);
    let inter_y1 = a_y1.max(b_y1);
    let inter_x2 = a_x2.min(b_x2);
    let inter_y2 = a_y2.min(b_y2);

    let inter_width = (inter_x2 - inter_x1).max(0.0);
    let inter_height = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_width * inter_height;

    let a_area = a.w * a.h;
    let b_area = b.w * b.h;
    let union_area = a_area + b_area - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a YOLO output tensor `[1, channels, n]` (channel-major) where
    /// prediction `i` has box (100, 100, 50, 60), face confidence `conf[i]`
    /// and every keypoint value set to `kpt`.
    fn tensor(channels: usize, conf: &[f32], kpt: f32) -> (Vec<usize>, Vec<f32>) {
        let n = conf.len();
        let mut data = vec![0.0f32; channels * n];
        for i in 0..n {
            data[i] = 100.0; // cx
            data[n + i] = 100.0; // cy
            data[2 * n + i] = 50.0; // w
            data[3 * n + i] = 60.0; // h
            data[4 * n + i] = conf[i];
            for c in 5..channels {
                data[c * n + i] = kpt;
            }
        }
        (vec![1, channels, n], data)
    }

    #[test]
    fn test_keypoint_values_are_not_read_as_confidence() {
        // YOLOv8-face layout: 4 box + 1 confidence + 15 keypoint values.
        // Keypoint coordinates are large pixel values; they must be ignored.
        let (shape, data) = tensor(20, &[0.9, 0.01, 0.6], 900.0);
        let dets = parse_yolo_output(&shape, &data, 0.5);

        let confs: Vec<f32> = dets.iter().map(|d| d.confidence).collect();
        assert_eq!(confs, vec![0.9, 0.6]);

        // ...but they are kept as the detection's 5 keypoints.
        assert_eq!(dets[0].keypoints.len(), 5);
        assert!(dets[0]
            .keypoints
            .iter()
            .all(|k| k.x == 900.0 && k.y == 900.0 && k.confidence == 900.0));
    }

    #[test]
    fn test_keypoint_channel_layout() {
        // One prediction, 2 keypoints: channels 5,6,7 then 8,9,10.
        let data = vec![
            100.0, 100.0, 50.0, 60.0, 0.9, // box + confidence
            11.0, 12.0, 0.8, // keypoint 0: x, y, visibility
            21.0, 22.0, 0.7, // keypoint 1
        ];
        let dets = parse_yolo_output(&[1, 11, 1], &data, 0.5);
        assert_eq!(
            dets[0].keypoints,
            vec![
                Keypoint {
                    x: 11.0,
                    y: 12.0,
                    confidence: 0.8
                },
                Keypoint {
                    x: 21.0,
                    y: 22.0,
                    confidence: 0.7
                },
            ]
        );
    }

    #[test]
    fn test_low_confidence_everywhere_gives_no_detections() {
        let (shape, data) = tensor(20, &[0.007, 0.001, 0.0], 908.0);
        assert!(parse_yolo_output(&shape, &data, 0.5).is_empty());
    }

    #[test]
    fn test_plain_box_plus_confidence_layout() {
        let (shape, data) = tensor(5, &[0.8, 0.2], 0.0);
        let dets = parse_yolo_output(&shape, &data, 0.5);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].confidence, 0.8);
    }

    #[test]
    fn test_too_few_channels_is_rejected() {
        let (shape, data) = tensor(4, &[], 0.0);
        assert!(parse_yolo_output(&shape, &data, 0.5).is_empty());
    }
}
