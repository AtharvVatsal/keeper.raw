use anyhow::{Context, Result};
use keeper_core::config::CullConfig;
use keeper_core::types::{ImageId, ImageRecord, ImageScore, RejectReason, Scene, Verdict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tracing::{debug, info, warn};

use crate::blink_detector::BlinkDetector;
use crate::face_detector::FaceDetector;
use crate::focus_scorer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CullManifest {
    pub total_images: usize,
    pub total_scenes: usize,
    pub total_keepers: usize,
    pub total_rejects: usize,
    pub total_unrated: usize,
    pub scenes: Vec<Scene>,
    pub image_scores: Vec<ImageScore>,
    pub processing_time_secs: f64,
}

pub struct CullPipeline {
    face_detector: Mutex<FaceDetector>,
    blink_detector: Mutex<BlinkDetector>,
    config: CullConfig,
}

impl CullPipeline {
    pub fn new(face_model_path: &Path, mesh_model_path: &Path, config: CullConfig) -> Result<Self> {
        info!("Initializing CullPipeline...");

        let face_detector =
            FaceDetector::new(face_model_path).context("Failed to load face detection model")?;
        info!("  ✓ Face detector loaded");

        let blink_detector =
            BlinkDetector::new(mesh_model_path).context("Failed to load face mesh model")?;
        info!("  ✓ Blink detector loaded");

        info!("CullPipeline ready.");

        Ok(CullPipeline {
            face_detector: Mutex::new(face_detector),
            blink_detector: Mutex::new(blink_detector),
            config,
        })
    }

    pub fn run(&self, images: &[ImageRecord]) -> Result<CullManifest> {
        let start_time = std::time::Instant::now();

        if images.is_empty() {
            return Ok(CullManifest {
                total_images: 0,
                total_scenes: 0,
                total_keepers: 0,
                total_rejects: 0,
                total_unrated: 0,
                scenes: vec![],
                image_scores: vec![],
                processing_time_secs: 0.0,
            });
        }

        info!("Starting cull pipeline on {} images...", images.len());

        info!("Stage 1/3: Stacking scenes...");
        let mut scenes = keeper_stacker::stacker::stack_scenes(images, &self.config)
            .context("Scene stacking failed")?;
        info!("  → {} scenes created", scenes.len());

        info!("Stage 2/3: Analyzing images (face detection + scoring + blink)...");

        let image_scores: Vec<ImageScore> = images
            .iter()
            .enumerate()
            .map(|(i, image)| {
                if (i + 1) % 100 == 0 || i + 1 == images.len() {
                    info!("  Processing image {}/{}", i + 1, images.len());
                }
                self.analyze_single_image(image)
            })
            .collect();

        let score_map: HashMap<ImageId, &ImageScore> =
            image_scores.iter().map(|s| (s.image_id, s)).collect();

        info!("Stage 3/3: Selecting keepers...");

        for scene in &mut scenes {
            let scene_scores: Vec<(u64, focus_scorer::FocusResult)> = scene
                .image_ids
                .iter()
                .filter_map(|id| {
                    score_map.get(id).map(|score| {
                        (
                            *id,
                            focus_scorer::FocusResult {
                                sharpness_score: score.sharpness_score,
                                raw_variance: score.sharpness_score,
                                noise_estimate: 1.0,
                                scored_region: focus_scorer::ScoredRegion::CenterFallback,
                            },
                        )
                    })
                })
                .collect();

            let non_blink_scores: Vec<(u64, focus_scorer::FocusResult)> = scene_scores
                .iter()
                .filter(|(id, _)| score_map.get(id).map(|s| !s.blink_detected).unwrap_or(true))
                .cloned()
                .collect();

            let keeper_id = if !non_blink_scores.is_empty() {
                focus_scorer::pick_keeper(&non_blink_scores)
            } else {
                focus_scorer::pick_keeper(&scene_scores)
            };

            scene.keeper_id = keeper_id;

            if let Some(kid) = keeper_id {
                debug!("Scene {}: keeper = image {}", scene.id, kid);
            }
        }

        let mut final_scores = image_scores;

        let keeper_ids: std::collections::HashSet<ImageId> =
            scenes.iter().filter_map(|s| s.keeper_id).collect();

        for score in &mut final_scores {
            if keeper_ids.contains(&score.image_id) {
                score.verdict = Verdict::Keeper;
            } else if score.blink_detected {
                score.verdict = Verdict::Reject {
                    reason: RejectReason::Blink,
                };
            }
        }

        let total_keepers = final_scores
            .iter()
            .filter(|s| matches!(s.verdict, Verdict::Keeper))
            .count();
        let total_rejects = final_scores
            .iter()
            .filter(|s| matches!(s.verdict, Verdict::Reject { .. }))
            .count();
        let total_unrated = final_scores
            .iter()
            .filter(|s| matches!(s.verdict, Verdict::Unrated))
            .count();

        let elapsed = start_time.elapsed().as_secs_f64();

        info!("Pipeline complete in {:.1}s:", elapsed);
        info!("  {} images → {} scenes", images.len(), scenes.len());
        info!(
            "  {} keepers, {} rejects, {} unrated",
            total_keepers, total_rejects, total_unrated
        );

        Ok(CullManifest {
            total_images: images.len(),
            total_scenes: scenes.len(),
            total_keepers,
            total_rejects,
            total_unrated,
            scenes,
            image_scores: final_scores,
            processing_time_secs: elapsed,
        })
    }

    fn analyze_single_image(&self, image: &ImageRecord) -> ImageScore {
        if image.preview_data.is_empty() {
            warn!(
                "Image '{}' has no preview data — skipping analysis",
                image.file_name
            );
            return ImageScore {
                image_id: image.id,
                sharpness_score: 0.0,
                has_face: false,
                face_bbox: None,
                blink_detected: false,
                eye_aspect_ratio: None,
                verdict: Verdict::Unrated,
            };
        }

        let faces = {
            let mut detector = self.face_detector.lock().unwrap();
            match detector.detect(
                &image.preview_data,
                self.config.face_detection_confidence,
                0.45,
            ) {
                Ok(f) => f,
                Err(e) => {
                    warn!("Face detection failed for '{}': {}", image.file_name, e);
                    vec![]
                }
            }
        };

        let primary_face = faces.first();
        let has_face = primary_face.is_some();
        let face_bbox = primary_face.map(|f| f.bbox);

        let sharpness_score =
            match focus_scorer::score_sharpness(&image.preview_data, face_bbox.as_ref()) {
                Ok(result) => result.sharpness_score,
                Err(e) => {
                    warn!("Focus scoring failed for '{}': {}", image.file_name, e);
                    0.0
                }
            };

        let (blink_detected, eye_aspect_ratio) = if let Some(ref bbox) = face_bbox {
            let mut detector = self.blink_detector.lock().unwrap();
            match detector.detect_blink(&image.preview_data, bbox, self.config.blink_ear_threshold)
            {
                Ok(result) => (result.blink_detected, Some(result.average_ear)),
                Err(e) => {
                    debug!(
                        "Blink detection failed for '{}': {} (non-fatal)",
                        image.file_name, e
                    );
                    (false, None)
                }
            }
        } else {
            (false, None)
        };

        ImageScore {
            image_id: image.id,
            sharpness_score,
            has_face,
            face_bbox,
            blink_detected,
            eye_aspect_ratio,
            verdict: Verdict::Unrated,
        }
    }
}
