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

        let (padded, transform) = preprocess::letterbox(&img, self.input_size, self.input_size);

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
                FaceDetection {
                    bbox: BoundingBox {
                        x,
                        y,
                        width: w,
                        height: h,
                    },
                    confidence: det.confidence,
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
}

fn parse_yolo_output(
    shape: &[usize],
    data: &[f32],
    confidence_threshold: f32,
) -> Vec<RawDetection> {
    if shape.len() < 3 {
        warn!("Unexpected YOLO output shape: {:?}", shape);
        return vec![];
    }

    let num_values = shape[1];
    let num_preds = shape[2];

    let num_classes = if num_values <= 4 { 1 } else { num_values - 4 };

    debug!(
        "YOLO output: {} values/det, {} predictions, {} class(es)",
        num_values, num_preds, num_classes
    );

    let mut detections = Vec::new();

    for i in 0..num_preds {
        let cx = data[i];
        let cy = data[num_preds + i];
        let w = data[num_preds * 2 + i];
        let h = data[num_preds * 3 + i];

        let confidence = if num_classes == 1 {
            data[4 * num_preds + i]
        } else {
            let mut max_score: f32 = 0.0;
            for c in 4..4 + num_classes {
                let score = data[c * num_preds + i];
                if score > max_score {
                    max_score = score;
                }
            }
            max_score
        };

        if confidence >= confidence_threshold {
            detections.push(RawDetection {
                cx,
                cy,
                w,
                h,
                confidence,
            });
        }
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
