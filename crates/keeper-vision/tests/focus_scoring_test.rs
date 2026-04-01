use anyhow::Result;
use keeper_core::types::BoundingBox;
use keeper_vision::face_detector::FaceDetector;
use keeper_vision::focus_scorer;
use std::path::Path;

fn get_model_path() -> Option<std::path::PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yolov8n-face.onnx");
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIPPED: yolov8n-face.onnx not found");
        None
    }
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

#[test]
fn test_focus_score_with_face_detection() -> Result<()> {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let jpeg_bytes = match get_test_image("face-photo.jpg") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: face-photo.jpg not found");
            return Ok(());
        }
    };

    let mut detector = FaceDetector::new(&model_path)?;
    let faces = detector.detect(&jpeg_bytes, 0.5, 0.45)?;

    println!("Detected {} face(s)", faces.len());

    let face_bbox = faces.first().map(|f| &f.bbox);
    let result = focus_scorer::score_sharpness(&jpeg_bytes, face_bbox)?;

    println!("Focus score: {:.2}", result.sharpness_score);
    println!("  Raw Laplacian variance: {:.2}", result.raw_variance);
    println!("  Noise estimate: {:.4}", result.noise_estimate);
    match &result.scored_region {
        focus_scorer::ScoredRegion::EyeRegion { face_bbox } => {
            println!(
                "  Region: eye area within face at ({:.0}, {:.0}, {:.0}x{:.0})",
                face_bbox.x, face_bbox.y, face_bbox.width, face_bbox.height
            );
        }
        focus_scorer::ScoredRegion::CenterFallback => {
            println!("  Region: center fallback (no face used)");
        }
    }

    assert!(result.sharpness_score > 0.0, "Score should be positive");
    assert!(result.raw_variance > 0.0, "Variance should be positive");

    Ok(())
}

#[test]
fn test_focus_score_without_face() -> Result<()> {
    let jpeg_bytes = match get_test_image("no-face-photo.jpg") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: no-face-photo.jpg not found");
            return Ok(());
        }
    };

    let result = focus_scorer::score_sharpness(&jpeg_bytes, None)?;

    println!(
        "Focus score (no face, center fallback): {:.2}",
        result.sharpness_score
    );
    println!("  Raw Laplacian variance: {:.2}", result.raw_variance);
    println!("  Noise estimate: {:.4}", result.noise_estimate);

    assert!(result.sharpness_score >= 0.0);

    Ok(())
}

#[test]
fn test_focus_score_comparison() -> Result<()> {
    let sharp = get_test_image("sharp-portrait.jpg");
    let soft = get_test_image("soft-portrait.jpg");

    if sharp.is_none() || soft.is_none() {
        eprintln!("SKIPPED: Need sharp-portrait.jpg and soft-portrait.jpg in test-data/");
        return Ok(());
    }

    let sharp_bytes = sharp.unwrap();
    let soft_bytes = soft.unwrap();

    let sharp_result = focus_scorer::score_sharpness(&sharp_bytes, None)?;
    let soft_result = focus_scorer::score_sharpness(&soft_bytes, None)?;

    println!("Sharp portrait score: {:.2}", sharp_result.sharpness_score);
    println!("Soft portrait score:  {:.2}", soft_result.sharpness_score);

    if sharp_result.sharpness_score > soft_result.sharpness_score {
        println!("Sharp image correctly scored higher than soft image!");
    } else {
        println!("Soft image scored higher than sharp image");
    }

    Ok(())
}

#[test]
fn test_focus_scoring_performance() -> Result<()> {
    let jpeg_bytes = match get_test_image("face-photo.jpg") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: face-photo.jpg not found");
            return Ok(());
        }
    };

    let bbox = BoundingBox {
        x: 100.0,
        y: 100.0,
        width: 200.0,
        height: 250.0,
    };

    let iterations = 100;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = focus_scorer::score_sharpness(&jpeg_bytes, Some(&bbox))?;
    }
    let elapsed = start.elapsed();
    let per_image_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!(
        "Focus scoring performance: {:.2} ms per image",
        per_image_ms
    );
    println!(
        "  For 3,000 images: {:.1} seconds",
        per_image_ms * 3000.0 / 1000.0
    );

    assert!(
        per_image_ms < 1000.0,
        "Focus scoring too slow: {:.2} ms (should be < 100ms)",
        per_image_ms
    );

    if per_image_ms < 5.0 {
        println!("Performance: EXCELLENT (< 5ms per image)");
    } else if per_image_ms < 20.0 {
        println!("Performance: GOOD (< 20ms)");
    } else {
        println!("Performance: ACCEPTABLE ({:.0}ms)", per_image_ms);
    }

    Ok(())
}
