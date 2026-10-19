use serde::{Deserialize, Serialize};

/// `#[serde(default)]`: fields missing from a saved config file (e.g. one
/// written by an older version) take their default value instead of making
/// the whole file fail to load. Unknown old fields are ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CullConfig {
    /// Max seconds between shots to count as the same burst.
    pub burst_threshold_secs: f64,

    /// How similar two images must be (perceptual hash distance).
    pub phash_similarity_threshold: u32,

    /// Both eyes' blink scores (0 = open, 1 = closed) must reach this for a
    /// shot to be rejected as a blink. Higher = stricter (fewer rejects).
    /// Replaced the old `blink_ear_threshold` (Eye Aspect Ratio) setting.
    pub blink_threshold: f64,

    /// How confident the face detector must be (0.0 to 1.0).
    pub face_detection_confidence: f32,
}

impl Default for CullConfig {
    fn default() -> Self {
        Self {
            burst_threshold_secs: 3.0,
            phash_similarity_threshold: 14,
            blink_threshold: 0.40,
            face_detection_confidence: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_old_config_with_ear_threshold_still_loads() {
        // A config.toml written before the blink-detector change.
        let old = "burst_threshold_secs = 2.5\n\
                   phash_similarity_threshold = 12\n\
                   blink_ear_threshold = 0.21\n\
                   face_detection_confidence = 0.6\n";
        let config: CullConfig = serde_json::from_value(toml_like(old)).unwrap();
        assert_eq!(config.burst_threshold_secs, 2.5);
        assert_eq!(config.phash_similarity_threshold, 12);
        assert_eq!(config.face_detection_confidence, 0.6);
        assert_eq!(
            config.blink_threshold, 0.40,
            "new field falls back to default"
        );
    }

    /// Minimal `key = number` parser so this test doesn't need the toml crate.
    fn toml_like(s: &str) -> serde_json::Value {
        let map = s
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| {
                let n: f64 = v.trim().parse().unwrap();
                let v = if n.fract() == 0.0 {
                    serde_json::json!(n as u64)
                } else {
                    serde_json::json!(n)
                };
                (k.trim().to_string(), v)
            })
            .collect();
        serde_json::Value::Object(map)
    }
}
