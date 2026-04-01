use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub type ImageId = u64;
pub type SceneId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: ImageId,
    pub file_path: PathBuf,
    pub file_name: String,
    pub timestamp: Option<i64>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub focal_length_mm: Option<f32>,
    pub aperture: Option<f32>,
    pub iso: Option<u32>,
    pub preview_width: u32,
    pub preview_height: u32,
    #[serde(skip)]
    pub preview_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageScore {
    pub image_id: ImageId,
    pub sharpness_score: f64,
    pub has_face: bool,
    pub face_bbox: Option<BoundingBox>,
    pub blink_detected: bool,
    pub eye_aspect_ratio: Option<f64>,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Keeper,
    Reject { reason: RejectReason },
    Unrated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    Blink,
    OutOfFocus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub id: SceneId,
    pub image_ids: Vec<ImageId>,
    pub keeper_id: Option<ImageId>,
}
