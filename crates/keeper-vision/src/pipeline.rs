use anyhow::{anyhow, Context, Result};
use keeper_core::config::CullConfig;
use keeper_core::types::{ImageId, ImageRecord, ImageScore, RejectReason, Scene, Verdict};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tracing::{debug, info, warn};

use keeper_stacker::hasher::{self, ImageHash};

use crate::blink_detector::BlinkDetector;
use crate::face_detector::{FaceDetection, FaceDetector};
use crate::focus_scorer;
use crate::preprocess;
use crate::quality;

/// Progress of a cull, reported after each analysed image. Analysis is
/// almost all of a cull's time; stacking and keeper selection take
/// milliseconds afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CullProgress {
    pub analysed: usize,
    pub total: usize,
}

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
    /// One set of models per worker thread; worker `i` only ever touches
    /// `analyzers[i]`. ONNX sessions need `&mut` to run, so a single shared
    /// set forced images through one at a time.
    analyzers: Vec<Mutex<Analyzer>>,
    config: CullConfig,
}

/// One image's analysis: its score and pHash (`None` if it couldn't be decoded).
type Analysis = (ImageScore, Option<ImageHash>);

/// One worker's own copy of the models.
struct Analyzer {
    face: FaceDetector,
    blink: BlinkDetector,
}

/// Default number of images analysed in parallel: half the logical cores
/// (each worker's models also use a couple of threads), at most 8.
pub fn default_workers() -> usize {
    (crate::runtime::available_cores() / 2).clamp(1, 8)
}

/// Face detection model file inside the models directory.
pub const FACE_MODEL_FILE: &str = "yolov8n-face.onnx";

/// Every model file the pipeline needs, relative to the models directory.
pub const REQUIRED_MODEL_FILES: [&str; 3] = [
    FACE_MODEL_FILE,
    crate::blink_detector::LANDMARK_MODEL_FILE,
    crate::blink_detector::BLENDSHAPE_MODEL_FILE,
];

/// Returns the required model files missing from `models_dir` (empty = all
/// present), so callers can show a helpful message before loading.
pub fn missing_model_files(models_dir: &Path) -> Vec<&'static str> {
    REQUIRED_MODEL_FILES
        .into_iter()
        .filter(|f| !models_dir.join(f).exists())
        .collect()
}

impl CullPipeline {
    /// Loads all models from `models_dir` (see [`REQUIRED_MODEL_FILES`]),
    /// with [`default_workers`] parallel workers.
    pub fn new(models_dir: &Path, config: CullConfig) -> Result<Self> {
        Self::with_workers(models_dir, config, default_workers())
    }

    /// Same as [`new`](Self::new) with an explicit number of parallel
    /// workers (1 = analyse images one at a time).
    pub fn with_workers(models_dir: &Path, config: CullConfig, workers: usize) -> Result<Self> {
        let workers = workers.max(1);
        // Split the cores between the workers' ONNX sessions.
        let threads_per_model = (crate::runtime::available_cores() / workers).max(1);
        info!(
            "Initializing CullPipeline from {:?}: {} worker(s), {} thread(s) per model...",
            models_dir, workers, threads_per_model
        );

        // Load each worker's models in parallel (each set takes ~0.3 s).
        let loaded: Vec<Result<Analyzer>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| -> Result<Analyzer> {
                        Ok(Analyzer {
                            face: FaceDetector::with_threads(
                                &models_dir.join(FACE_MODEL_FILE),
                                threads_per_model,
                            )
                            .context("Failed to load face detection model")?,
                            blink: BlinkDetector::with_threads(models_dir, threads_per_model)
                                .context("Failed to load blink detection models")?,
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(anyhow!("model loading thread panicked")))
                })
                .collect()
        });
        let analyzers = loaded
            .into_iter()
            .map(|a| a.map(Mutex::new))
            .collect::<Result<Vec<_>>>()?;

        info!("CullPipeline ready.");
        Ok(CullPipeline { analyzers, config })
    }

    /// Number of images analysed in parallel.
    pub fn workers(&self) -> usize {
        self.analyzers.len()
    }

    pub fn run(&self, images: &[ImageRecord]) -> Result<CullManifest> {
        self.run_with_progress(images, &|_| {})
    }

    /// Same as [`run`](Self::run), calling `on_progress` after each image is
    /// analysed. It is called from the worker threads, possibly at the same
    /// time, so it must be cheap and thread-safe.
    pub fn run_with_progress(
        &self,
        images: &[ImageRecord],
        on_progress: &(dyn Fn(CullProgress) + Sync),
    ) -> Result<CullManifest> {
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

        // Analysis comes first: each preview is decoded ONCE and every stage
        // (pHash, face, focus, blink) works on that decode. Stacking then only
        // needs timestamps + the stored hashes. Images are analysed in
        // parallel, one per worker, so at most `workers` decoded previews are
        // in memory at once. Results come back in input order.
        info!(
            "Stage 1/3: Analyzing images (decode once → pHash, face, focus, blink) with {} worker(s)...",
            self.workers()
        );

        // Plain worker threads (not a work-stealing pool): each one owns its
        // analyzer for the whole run and pulls the next image index from a
        // shared counter. A work-stealing pool deadlocked here: a worker
        // holding its analyzer could steal another image's task while waiting
        // on nested parallel work, and that task then waited on the same lock.
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let per_worker: Vec<Vec<(usize, Analysis)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .analyzers
                .iter()
                .map(|slot| {
                    let (next, done) = (&next, &done);
                    scope.spawn(move || {
                        let mut analyzer =
                            slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(image) = images.get(i) else {
                                break;
                            };
                            out.push((i, self.analyze_single_image(&mut analyzer, image)));

                            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                            on_progress(CullProgress {
                                analysed: n,
                                total: images.len(),
                            });
                            if n.is_multiple_of(100) || n == images.len() {
                                info!("  Analysed {}/{} images", n, images.len());
                            }
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("analysis worker panicked"))
                .collect()
        });

        // Put results back in input order.
        let mut analysed: Vec<(usize, Analysis)> = per_worker.into_iter().flatten().collect();
        analysed.sort_unstable_by_key(|(i, _)| *i);
        let analysed = analysed.into_iter().map(|(_, result)| result);

        let mut hashes: HashMap<ImageId, ImageHash> = HashMap::with_capacity(images.len());
        let mut image_scores: Vec<ImageScore> = Vec::with_capacity(images.len());
        for (score, hash) in analysed {
            if let Some(hash) = hash {
                hashes.insert(score.image_id, hash);
            }
            image_scores.push(score);
        }

        info!("Stage 2/3: Stacking scenes...");
        let mut scenes =
            keeper_stacker::stacker::stack_scenes_with_hashes(images, &hashes, &self.config)
                .context("Scene stacking failed")?;
        info!("  → {} scenes created", scenes.len());

        info!("Stage 3/3: Selecting keepers...");
        let mut final_scores = image_scores;
        select_keepers(&mut scenes, &mut final_scores);

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

    /// Analyses one image from a single decode. Returns its score and its
    /// pHash (`None` if the preview couldn't be decoded).
    fn analyze_single_image(&self, analyzer: &mut Analyzer, image: &ImageRecord) -> Analysis {
        let unscored = ImageScore {
            image_id: image.id,
            sharpness_score: 0.0,
            has_face: false,
            face_bbox: None,
            blink_detected: false,
            blink_score: None,
            peak_sharpness: 0.0,
            verdict: Verdict::Unrated,
        };

        // In memory, or read from disk now — only while this image is being
        // analysed, so at most one preview per worker is in RAM.
        let preview = match image.load_preview() {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("Could not read preview for '{}': {}", image.file_name, e);
                return (unscored, None);
            }
        };
        if preview.is_empty() {
            warn!(
                "Image '{}' has no preview data — skipping analysis",
                image.file_name
            );
            return (unscored, None);
        }

        // ---- Decode once ----------------------------------------------------
        let decoded = match preprocess::decode_for_analysis(&preview, image.orientation) {
            Ok(d) => d,
            Err(e) => {
                warn!("Could not decode preview for '{}': {}", image.file_name, e);
                return (unscored, None);
            }
        };

        // ---- pHash (on the small copy) --------------------------------------
        let phash = hasher::compute_phash_image(&decoded.small);

        // ---- Unusable frame? (black / blank) ----------------------------------
        // Rejected outright and never chosen as a keeper; no point running
        // face, focus or blink analysis on it.
        if quality::is_unusable(&decoded.small) {
            debug!("'{}' is unusable (black or blank frame)", image.file_name);
            let score = ImageScore {
                verdict: Verdict::Reject {
                    reason: RejectReason::Unusable,
                },
                ..unscored
            };
            return (score, Some(phash));
        }

        // ---- Out of focus everywhere? (camera shake / missed focus) -----------
        // Rejected outright and never chosen as a keeper — not even when it is
        // the only photo of its moment.
        let peak_sharpness = quality::peak_sharpness(&decoded.work) as f64;
        if peak_sharpness < quality::BLURRY_PEAK_SHARPNESS as f64 {
            debug!(
                "'{}' is out of focus (peak sharpness {:.0})",
                image.file_name, peak_sharpness
            );
            let score = ImageScore {
                peak_sharpness,
                verdict: Verdict::Reject {
                    reason: RejectReason::OutOfFocus,
                },
                ..unscored
            };
            return (score, Some(phash));
        }

        // ---- Face detection (on the small copy) -----------------------------
        let faces = {
            let detector = &mut analyzer.face;
            match detector.detect_image(&decoded.small, self.config.face_detection_confidence, 0.45)
            {
                Ok(f) => f,
                Err(e) => {
                    warn!("Face detection failed for '{}': {}", image.file_name, e);
                    vec![]
                }
            }
        };

        let primary_face = faces.first();
        let has_face = primary_face.is_some();
        // Stored boxes are in preview coordinates (the loupe zooms on them);
        // focus and blink need the same box in `work` coordinates.
        let face_bbox =
            primary_face.map(|f| preprocess::scale_bbox(&f.bbox, decoded.small_to_preview()));
        let work_bbox =
            face_bbox.map(|b| preprocess::scale_bbox(&b, 1.0 / decoded.work_to_preview()));

        // ---- Focus (on the work image) ---------------------------------------
        let gray = decoded.work.to_luma8();
        let sharpness_score = match focus_scorer::score_sharpness_gray(&gray, work_bbox.as_ref()) {
            Ok(result) => result.sharpness_score,
            Err(e) => {
                warn!("Focus scoring failed for '{}': {}", image.file_name, e);
                0.0
            }
        };

        // ---- Blink (on the work image) ---------------------------------------
        // Head tilt from the two eye keypoints (an angle, so the same in
        // `small` and `work` coordinates). Used to level the face crop.
        let roll = primary_face.map(head_roll).unwrap_or(0.0);
        let (blink_detected, blink_score) = if let Some(ref bbox) = work_bbox {
            let detector = &mut analyzer.blink;
            match detector.detect_blink_image(
                &decoded.work,
                bbox,
                roll,
                self.config.blink_threshold,
            ) {
                Ok(result) => (result.blink_detected, Some(result.blink_score as f64)),
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

        let score = ImageScore {
            image_id: image.id,
            sharpness_score,
            has_face,
            face_bbox,
            blink_detected,
            blink_score,
            peak_sharpness,
            verdict: Verdict::Unrated,
        };
        (score, Some(phash))
    }
}

/// Picks each scene's keeper and sets every image's verdict.
///
/// - Frames rejected during analysis (black/blank `Unusable`, or
///   `OutOfFocus`) keep their verdict and are never a keeper.
/// - The keeper is the sharpest frame that isn't a clear blink. If every
///   usable frame blinked (including a lone frame), the scene has **no**
///   keeper: clear blinks are always rejected (av's reject policy).
/// - Blinks are `Reject { Blink }`; everything else is `Unrated`.
pub fn select_keepers(scenes: &mut [Scene], scores: &mut [ImageScore]) {
    let score_map: HashMap<ImageId, &ImageScore> = scores.iter().map(|s| (s.image_id, s)).collect();

    for scene in scenes.iter_mut() {
        let candidates: Vec<&ImageScore> = scene
            .image_ids
            .iter()
            .filter_map(|id| score_map.get(id).copied())
            .filter(|s| !is_rejected_in_analysis(s) && !s.blink_detected)
            .collect();

        scene.keeper_id = sharpest(&candidates);
        if let Some(kid) = scene.keeper_id {
            debug!("Scene {}: keeper = image {}", scene.id, kid);
        }
    }

    let keeper_ids: std::collections::HashSet<ImageId> =
        scenes.iter().filter_map(|s| s.keeper_id).collect();
    for score in scores.iter_mut() {
        if is_rejected_in_analysis(score) {
            continue; // keep its Unusable / OutOfFocus verdict
        }
        if keeper_ids.contains(&score.image_id) {
            score.verdict = Verdict::Keeper;
        } else if score.blink_detected {
            score.verdict = Verdict::Reject {
                reason: RejectReason::Blink,
            };
        }
    }
}

/// The id of the sharpest image, if any (ties: the last one, as `max_by` does).
fn sharpest(candidates: &[&ImageScore]) -> Option<ImageId> {
    candidates
        .iter()
        .max_by(|a, b| {
            a.sharpness_score
                .partial_cmp(&b.sharpness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.image_id)
}

/// True if analysis already rejected this image (black/blank frame or out
/// of focus everywhere). Such images are never a keeper.
fn is_rejected_in_analysis(score: &ImageScore) -> bool {
    matches!(
        score.verdict,
        Verdict::Reject {
            reason: RejectReason::Unusable | RejectReason::OutOfFocus
        }
    )
}

/// Head roll in radians from the two eye keypoints (0 if they're missing).
/// Positive = clockwise in image coordinates.
fn head_roll(face: &FaceDetection) -> f32 {
    match (face.keypoints.first(), face.keypoints.get(1)) {
        (Some(a), Some(b)) => (b.y - a.y).atan2(b.x - a.x),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_detector::Keypoint;
    use keeper_core::types::BoundingBox;

    fn face_with_eyes(a: (f32, f32), b: (f32, f32)) -> FaceDetection {
        let kp = |(x, y)| Keypoint {
            x,
            y,
            confidence: 0.9,
        };
        FaceDetection {
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
            confidence: 0.9,
            keypoints: vec![kp(a), kp(b)],
        }
    }

    fn score(id: ImageId, sharpness: f64, blink: bool) -> ImageScore {
        ImageScore {
            image_id: id,
            sharpness_score: sharpness,
            has_face: blink,
            face_bbox: None,
            blink_detected: blink,
            blink_score: None,
            peak_sharpness: 1000.0,
            verdict: Verdict::Unrated,
        }
    }

    fn scene(id: u64, ids: &[ImageId]) -> Scene {
        Scene {
            id,
            image_ids: ids.to_vec(),
            keeper_id: None,
        }
    }

    const BLINK: Verdict = Verdict::Reject {
        reason: RejectReason::Blink,
    };

    #[test]
    fn test_lone_sharp_photo_is_keeper() {
        let mut scenes = vec![scene(1, &[1])];
        let mut scores = vec![score(1, 50.0, false)];
        select_keepers(&mut scenes, &mut scores);
        assert_eq!(scenes[0].keeper_id, Some(1));
        assert_eq!(scores[0].verdict, Verdict::Keeper);
    }

    #[test]
    fn test_blink_never_beats_open_eyes() {
        let mut scenes = vec![scene(1, &[1, 2])];
        let mut scores = vec![score(1, 100.0, true), score(2, 10.0, false)];
        select_keepers(&mut scenes, &mut scores);
        assert_eq!(scenes[0].keeper_id, Some(2));
        assert_eq!(scores[0].verdict, BLINK);
    }

    #[test]
    fn test_lone_blink_is_rejected_not_kept() {
        // A clear blink with no alternative is still a blink: reject it.
        let mut scenes = vec![scene(1, &[1])];
        let mut scores = vec![score(1, 50.0, true)];
        select_keepers(&mut scenes, &mut scores);
        assert_eq!(scenes[0].keeper_id, None);
        assert_eq!(scores[0].verdict, BLINK);
    }

    #[test]
    fn test_scene_where_everyone_blinked_has_no_keeper() {
        let mut scenes = vec![scene(1, &[1, 2])];
        let mut scores = vec![score(1, 50.0, true), score(2, 80.0, true)];
        select_keepers(&mut scenes, &mut scores);
        assert_eq!(scenes[0].keeper_id, None);
        assert!(scores.iter().all(|s| s.verdict == BLINK));
    }

    #[test]
    fn test_unusable_frame_is_never_keeper() {
        let unusable = Verdict::Reject {
            reason: RejectReason::Unusable,
        };
        let mut scenes = vec![scene(1, &[1, 2])];
        let mut scores = vec![score(1, 900.0, false), score(2, 10.0, false)];
        scores[0].verdict = unusable;
        select_keepers(&mut scenes, &mut scores);
        assert_eq!(scenes[0].keeper_id, Some(2));
        assert_eq!(scores[0].verdict, unusable);
    }

    #[test]
    fn test_out_of_focus_frame_is_never_keeper() {
        let blurry = Verdict::Reject {
            reason: RejectReason::OutOfFocus,
        };
        let mut scenes = vec![scene(1, &[1]), scene(2, &[2, 3])];
        let mut scores = vec![
            score(1, 50.0, false),
            score(2, 90.0, false),
            score(3, 10.0, false),
        ];
        scores[0].verdict = blurry;
        scores[1].verdict = blurry;
        select_keepers(&mut scenes, &mut scores);
        // A lone blurry photo has no keeper; a blurry frame never beats a sharp one.
        assert_eq!(scenes[0].keeper_id, None);
        assert_eq!(scenes[1].keeper_id, Some(3));
        assert_eq!(scores[0].verdict, blurry);
        assert_eq!(scores[1].verdict, blurry);
    }

    #[test]
    fn test_head_roll() {
        assert_eq!(head_roll(&face_with_eyes((10.0, 50.0), (30.0, 50.0))), 0.0);
        let tilted = head_roll(&face_with_eyes((10.0, 50.0), (30.0, 70.0)));
        assert!((tilted - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        let mut no_kp = face_with_eyes((0.0, 0.0), (0.0, 0.0));
        no_kp.keypoints.clear();
        assert_eq!(head_roll(&no_kp), 0.0);
    }
}
