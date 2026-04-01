use anyhow::Result;
use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_vision::pipeline::CullPipeline;
use std::path::{Path, PathBuf};

fn get_model_paths() -> Option<(PathBuf, PathBuf)> {
    let face = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yolov8n-face.onnx");
    let mesh = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/face_landmark.onnx");

    if !face.exists() {
        eprintln!("SKIPPED: yolov8n-face.onnx not found");
        return None;
    }
    if !mesh.exists() {
        eprintln!("SKIPPED: face_landmark.onnx not found");
        return None;
    }

    Some((face, mesh))
}

fn load_test_jpeg(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name);
    if path.exists() {
        Some(std::fs::read(&path).unwrap())
    } else {
        None
    }
}

fn make_record(id: u64, name: &str, timestamp_ms: Option<i64>, data: Vec<u8>) -> ImageRecord {
    ImageRecord {
        id,
        file_path: PathBuf::from(name),
        file_name: name.to_string(),
        timestamp: timestamp_ms,
        camera_make: None,
        camera_model: None,
        focal_length_mm: None,
        aperture: None,
        iso: None,
        preview_width: 0,
        preview_height: 0,
        preview_data: data,
    }
}

#[test]
fn test_pipeline_with_burst_grouping() -> Result<()> {
    let (face_path, mesh_path) = match get_model_paths() {
        Some(p) => p,
        None => return Ok(()),
    };

    let face_jpg = match load_test_jpeg("face-photo.jpg") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: face-photo.jpg not found");
            return Ok(());
        }
    };

    let burst_a1 = load_test_jpeg("burst_a_01.jpg").unwrap_or_else(|| face_jpg.clone());
    let burst_a2 = load_test_jpeg("burst_a_02.jpg").unwrap_or_else(|| face_jpg.clone());

    let images = vec![
        make_record(1, "burst_01.jpg", Some(1000), burst_a1.clone()),
        make_record(2, "burst_02.jpg", Some(1200), burst_a2.clone()),
        make_record(3, "burst_03.jpg", Some(1400), face_jpg.clone()),
        make_record(4, "solo_shot.jpg", Some(6000), face_jpg.clone()),
    ];

    let config = CullConfig::default();
    let pipeline = CullPipeline::new(&face_path, &mesh_path, config)?;

    let manifest = pipeline.run(&images)?;

    println!("Pipeline results:");
    println!("  Images: {}", manifest.total_images);
    println!("  Scenes: {}", manifest.total_scenes);
    println!("  Keepers: {}", manifest.total_keepers);
    println!("  Rejects: {}", manifest.total_rejects);
    println!("  Time: {:.1}s", manifest.processing_time_secs);

    assert_eq!(manifest.total_images, 4);
    assert!(
        manifest.total_scenes >= 2 && manifest.total_scenes <= 4,
        "Expected 2-4 scenes, got {}",
        manifest.total_scenes
    );
    for scene in &manifest.scenes {
        assert!(
            scene.keeper_id.is_some(),
            "Scene {} should have a keeper",
            scene.id
        );
    }

    assert_eq!(
        manifest.total_keepers, manifest.total_scenes,
        "Each scene should produce one keeper"
    );

    println!("\n--- Scene details ---");
    for scene in &manifest.scenes {
        println!(
            "  Scene {}: {} images, keeper={}",
            scene.id,
            scene.image_ids.len(),
            scene.keeper_id.unwrap_or(0)
        );
    }

    Ok(())
}

#[test]
fn test_manifest_is_valid_json() -> Result<()> {
    let (face_path, mesh_path) = match get_model_paths() {
        Some(p) => p,
        None => return Ok(()),
    };

    let face_jpg = match load_test_jpeg("face-photo.jpg") {
        Some(b) => b,
        None => return Ok(()),
    };

    let images = vec![make_record(1, "test.jpg", Some(1000), face_jpg)];
    let config = CullConfig::default();
    let pipeline = CullPipeline::new(&face_path, &mesh_path, config)?;
    let manifest = pipeline.run(&images)?;

    let json = serde_json::to_string_pretty(&manifest)?;

    let _: serde_json::Value = serde_json::from_str(&json)?;

    println!("Manifest JSON ({} bytes):", json.len());
    println!("{}", &json[..json.len().min(500)]);

    Ok(())
}
