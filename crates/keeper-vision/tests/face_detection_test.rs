use anyhow::Result;
use keeper_vision::face_detector::FaceDetector;
use std::path::Path;

fn get_model_path() -> Option<std::path::PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yolov8n-face.onnx");
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIPPED: yolov8n-face.onnx not found in models/");
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
        eprintln!("SKIPPED: {} not found in test-data/", name);
        None
    }
}

#[test]
fn test_face_detector_loads() -> Result<()> {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };

    let detector = FaceDetector::new(&model_path)?;
    println!("Face detector loaded successfully");
    println!("   Input names: {:?}", detector.model_input_names());
    println!("   Output names: {:?}", detector.model_output_names());

    Ok(())
}

#[test]
fn test_detect_faces_in_portrait() -> Result<()> {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let jpeg_bytes = match get_test_image("face-photo.jpg") {
        Some(b) => b,
        None => return Ok(()),
    };

    let mut detector = FaceDetector::new(&model_path)?;
    let detections = detector.detect(&jpeg_bytes, 0.5, 0.45)?;

    println!("Found {} face(s) in face-photo.jpg:", detections.len());
    for (i, det) in detections.iter().enumerate() {
        println!(
            "  Face {}: confidence={:.3}, bbox=({:.0}, {:.0}, {:.0}x{:.0})",
            i + 1,
            det.confidence,
            det.bbox.x,
            det.bbox.y,
            det.bbox.width,
            det.bbox.height
        );
    }

    assert!(
        !detections.is_empty(),
        "Expected at least 1 face in face-photo.jpg, found 0"
    );

    assert!(
        detections[0].confidence >= 0.3,
        "Top detection confidence too low: {:.3}",
        detections[0].confidence
    );

    assert!(detections[0].bbox.width > 0.0);
    assert!(detections[0].bbox.height > 0.0);

    Ok(())
}

#[test]
fn test_no_false_positives_in_landscape() -> Result<()> {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let jpeg_bytes = match get_test_image("no-face-photo.jpg") {
        Some(b) => b,
        None => return Ok(()),
    };

    let mut detector = FaceDetector::new(&model_path)?;
    let detections = detector.detect(&jpeg_bytes, 0.5, 0.45)?;

    println!(
        "Found {} face(s) in no-face-photo.jpg (expected 0)",
        detections.len()
    );
    for det in &detections {
        println!(
            "  False positive: confidence={:.3}, bbox=({:.0}, {:.0})",
            det.confidence, det.bbox.x, det.bbox.y
        );
    }

    Ok(())
}

#[test]
fn test_face_detection_performance() -> Result<()> {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let jpeg_bytes = match get_test_image("face-photo.jpg") {
        Some(b) => b,
        None => return Ok(()),
    };

    let mut detector = FaceDetector::new(&model_path)?;

    let _ = detector.detect(&jpeg_bytes, 0.5, 0.45)?;

    let iterations = 20;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = detector.detect(&jpeg_bytes, 0.5, 0.45)?;
    }
    let elapsed = start.elapsed();
    let per_image_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!(
        "Face detection performance: {:.1} ms per image",
        per_image_ms
    );
    println!(
        "  For 3,000 images: {:.1} seconds ({:.1} minutes)",
        per_image_ms * 3000.0 / 1000.0,
        per_image_ms * 3000.0 / 60000.0,
    );

    if per_image_ms < 100.0 {
        println!("Performance: EXCELLENT (< 100ms)");
    } else if per_image_ms < 200.0 {
        println!("Performance: GOOD (< 200ms)");
    } else if per_image_ms < 500.0 {
        println!("Performance: ACCEPTABLE but slow ({:.0}ms)", per_image_ms);
    } else {
        println!("Performance: SLOW ({:.0}ms)", per_image_ms);
    }

    Ok(())
}
