use anyhow::Result;
use keeper_vision::blink_detector::{BlinkDetector, BlinkResult};
use keeper_vision::face_detector::FaceDetector;
use keeper_vision::{pipeline, preprocess};
use std::path::{Path, PathBuf};

fn get_models_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let missing = pipeline::missing_model_files(&dir);
    if !missing.is_empty() {
        eprintln!("SKIPPED: missing model file(s): {}", missing.join(", "));
        return None;
    }
    Some(dir)
}

fn get_test_image(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name);
    if path.exists() {
        Some(std::fs::read(&path).unwrap())
    } else {
        None
    }
}

/// Runs blink detection the way the pipeline does: upright image, face on
/// the small copy, levelled crop on the work image.
fn blink_on(models_dir: &Path, jpeg: &[u8]) -> Result<Option<BlinkResult>> {
    let decoded = preprocess::decode_for_analysis(jpeg, preprocess::jpeg_orientation(jpeg))?;
    let mut faces = FaceDetector::new(&models_dir.join(pipeline::FACE_MODEL_FILE))?;
    let Some(face) = faces
        .detect_image(&decoded.small, 0.5, 0.45)?
        .into_iter()
        .next()
    else {
        return Ok(None);
    };
    let to_work = decoded.work.width() as f32 / decoded.small.width() as f32;
    let bbox = preprocess::scale_bbox(&face.bbox, to_work);
    let roll = match (face.keypoints.first(), face.keypoints.get(1)) {
        (Some(a), Some(b)) => (b.y - a.y).atan2(b.x - a.x),
        _ => 0.0,
    };
    let mut blink = BlinkDetector::new(models_dir)?;
    Ok(Some(blink.detect_blink_image(
        &decoded.work,
        &bbox,
        roll,
        0.40,
    )?))
}

#[test]
fn test_blink_detector_loads() -> Result<()> {
    let Some(dir) = get_models_dir() else {
        return Ok(());
    };
    BlinkDetector::new(&dir)?;
    Ok(())
}

#[test]
fn test_open_eyes_portrait_is_not_a_blink() -> Result<()> {
    let Some(dir) = get_models_dir() else {
        return Ok(());
    };
    let Some(jpeg) = get_test_image("face-photo.JPG") else {
        eprintln!("SKIPPED: face-photo.JPG not found");
        return Ok(());
    };

    let result = blink_on(&dir, &jpeg)?.expect("face-photo.JPG should contain a face");
    println!("{result:?}");

    for score in [
        result.left_blink,
        result.right_blink,
        result.face_confidence,
    ] {
        assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
    }
    assert!(
        result.face_confidence >= 0.5,
        "FaceMesh should be confident on an upright portrait: {:.2}",
        result.face_confidence
    );
    assert!(
        !result.blink_detected,
        "open eyes flagged as a blink: {result:?}"
    );
    Ok(())
}

#[test]
#[ignore = "timing-based; run locally with: cargo test --release -- --ignored"]
fn test_blink_detection_performance() -> Result<()> {
    let Some(dir) = get_models_dir() else {
        return Ok(());
    };
    let Some(jpeg) = get_test_image("face-photo.JPG") else {
        return Ok(());
    };
    let decoded = preprocess::decode_for_analysis(&jpeg, preprocess::jpeg_orientation(&jpeg))?;
    let mut faces = FaceDetector::new(&dir.join(pipeline::FACE_MODEL_FILE))?;
    let Some(face) = faces
        .detect_image(&decoded.small, 0.5, 0.45)?
        .into_iter()
        .next()
    else {
        return Ok(());
    };
    let bbox = preprocess::scale_bbox(
        &face.bbox,
        decoded.work.width() as f32 / decoded.small.width() as f32,
    );
    let mut blink = BlinkDetector::new(&dir)?;
    blink.detect_blink_image(&decoded.work, &bbox, 0.0, 0.40)?;

    let iterations = 20;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        blink.detect_blink_image(&decoded.work, &bbox, 0.0, 0.40)?;
    }
    let per_face_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Blink detection: {per_face_ms:.1} ms per face");
    assert!(
        per_face_ms < 1000.0,
        "blink detection too slow: {per_face_ms:.1} ms"
    );
    Ok(())
}
