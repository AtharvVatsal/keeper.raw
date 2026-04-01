use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CullConfig {
    /// Max seconds between shots to count as the same burst.
    pub burst_threshold_secs: f64,

    /// How similar two images must be (perceptual hash distance).
    pub phash_similarity_threshold: u32,

    /// Eye openness below this = blink.
    pub blink_ear_threshold: f64,

    /// How confident the face detector must be (0.0 to 1.0).
    pub face_detection_confidence: f32,
}

impl Default for CullConfig {
    fn default() -> Self {
        Self {
            burst_threshold_secs: 3.0,
            phash_similarity_threshold: 14,
            blink_ear_threshold: 0.21,
            face_detection_confidence: 0.5,
        }
    }
}