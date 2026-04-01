//! keeper-stacker: Scene grouping (burst detection) for keeper.raw
//!
//! Groups rapid-fire burst shots into "Scenes" so the vision pipeline
//! can pick the best image from each group rather than scoring all
//! 3,000 images independently.
//!
//! Two-stage algorithm:
//! 1. Timestamp clustering — consecutive shots within a time threshold
//! 2. pHash sub-splitting — splits clusters where composition changed

pub mod hasher;
pub mod stacker;