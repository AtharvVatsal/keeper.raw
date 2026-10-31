//! Computer vision pipeline for keeper.raw.
//!
//! Handles ML inference: face detection, sharpness scoring, blink detection
//! and accidental-shot detection. Uses ONNX Runtime on CPU.

pub mod accidental;
pub mod blink_detector;
pub mod face_detector;
pub mod focus_scorer;
pub mod pipeline;
pub mod preprocess;
pub mod quality;
pub mod runtime;
