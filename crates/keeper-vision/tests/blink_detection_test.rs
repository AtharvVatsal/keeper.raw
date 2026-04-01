use anyhow::Result;
use keeper_vision::blink_detector::BlinkDetector;
use keeper_vision::face_detector::FaceDetector;
use std::path::Path;

fn get_face_model_path() -> Option<std::path::PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yolov8n-face.onnx");
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIPPED: yolov8n-face.onnx not found");
        None
    }
}

fn get_mesh_model_path() -> Option<std::path::PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/face_landmark.onnx");
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIPPED: face_landmark.onnx not found in models/");
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
fn test_blink_detector_loads() -> Result<()> {
    let mesh_path = match get_mesh_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };

    let _detector = BlinkDetector::new(&mesh_path)?;
    println!("✅ Blink detector loaded successfully");

    Ok(())
}

#[test]
fn test_blink_detection_on_portrait() -> Result<()> {
    let face_model_path = match get_face_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let mesh_model_path = match get_mesh_model_path() {
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

    let mut face_detector = FaceDetector::new(&face_model_path)?;
    let faces = face_detector.detect(&jpeg_bytes, 0.5, 0.45)?;

    if faces.is_empty() {
        eprintln!("No faces detected in face-photo.jpg — can't test blink detection");
        return Ok(());
    }

    println!(
        "Detected {} face(s), testing blink on first face",
        faces.len()
    );

    let mut blink_detector = BlinkDetector::new(&mesh_model_path)?;
    let result = blink_detector.detect_blink(&jpeg_bytes, &faces[0].bbox, 0.21)?;

    println!("Blink detection result:");
    println!("  Right EAR: {:.3}", result.right_ear);
    println!("  Left EAR:  {:.3}", result.left_ear);
    println!("  Average EAR: {:.3}", result.average_ear);
    println!("  Blink detected: {}", result.blink_detected);

    if !result.blink_detected {
        println!("No blink detected in normal portrait (correct!)");
    } else {
        println!("Blink detected in normal portrait");
    }

    assert!(
        result.average_ear >= 0.0 && result.average_ear <= 1.0,
        "EAR value out of expected range: {:.3}",
        result.average_ear
    );

    Ok(())
}

#[test]
fn test_blink_detection_performance() -> Result<()> {
    let face_model_path = match get_face_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let mesh_model_path = match get_mesh_model_path() {
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

    let mut face_detector = FaceDetector::new(&face_model_path)?;
    let faces = face_detector.detect(&jpeg_bytes, 0.5, 0.45)?;

    if faces.is_empty() {
        eprintln!("No faces detected — skipping performance test");
        return Ok(());
    }

    let mut blink_detector = BlinkDetector::new(&mesh_model_path)?;

    let _ = blink_detector.detect_blink(&jpeg_bytes, &faces[0].bbox, 0.21)?;

    let iterations = 10;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = blink_detector.detect_blink(&jpeg_bytes, &faces[0].bbox, 0.21)?;
    }
    let elapsed = start.elapsed();
    let per_image_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!(
        "Blink detection performance: {:.1} ms per face",
        per_image_ms
    );
    println!(
        "  For 3,000 images (assuming 1 face each): {:.1} seconds ({:.1} minutes)",
        per_image_ms * 3000.0 / 1000.0,
        per_image_ms * 3000.0 / 60000.0,
    );

    if per_image_ms < 50.0 {
        println!("Performance: EXCELLENT (< 50ms)");
    } else if per_image_ms < 200.0 {
        println!("Performance: GOOD (< 200ms)");
    } else if per_image_ms < 1000.0 {
        println!("Performance: ACCEPTABLE ({:.0}ms)", per_image_ms);
    } else {
        println!("Performance: SLOW ({:.0}ms)", per_image_ms);
    }

    Ok(())
}
