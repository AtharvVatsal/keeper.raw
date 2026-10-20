use anyhow::Result;
use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_vision::pipeline::CullPipeline;
use std::path::{Path, PathBuf};

fn get_models_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let missing = keeper_vision::pipeline::missing_model_files(&dir);
    if !missing.is_empty() {
        eprintln!("SKIPPED: missing model file(s): {}", missing.join(", "));
        return None;
    }
    Some(dir)
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
        orientation: keeper_vision::preprocess::jpeg_orientation(&data),
        preview_data: data,
        preview_file: None,
    }
}

#[test]
fn test_pipeline_with_burst_grouping() -> Result<()> {
    let models_dir = match get_models_dir() {
        Some(p) => p,
        None => return Ok(()),
    };

    let face_jpg = match load_test_jpeg("face-photo.JPG") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: face-photo.JPG not found");
            return Ok(());
        }
    };

    let burst_a1 = load_test_jpeg("burst_a_01.JPG").unwrap_or_else(|| face_jpg.clone());
    let burst_a2 = load_test_jpeg("burst_a_02.JPG").unwrap_or_else(|| face_jpg.clone());

    let images = vec![
        make_record(1, "burst_01.jpg", Some(1000), burst_a1.clone()),
        make_record(2, "burst_02.jpg", Some(1200), burst_a2.clone()),
        make_record(3, "burst_03.jpg", Some(1400), face_jpg.clone()),
        make_record(4, "solo_shot.jpg", Some(6000), face_jpg.clone()),
    ];

    let config = CullConfig::default();
    let pipeline = CullPipeline::new(&models_dir, config)?;

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
    let models_dir = match get_models_dir() {
        Some(p) => p,
        None => return Ok(()),
    };

    let face_jpg = match load_test_jpeg("face-photo.JPG") {
        Some(b) => b,
        None => return Ok(()),
    };

    let images = vec![make_record(1, "test.jpg", Some(1000), face_jpg)];
    let config = CullConfig::default();
    let pipeline = CullPipeline::new(&models_dir, config)?;
    let manifest = pipeline.run(&images)?;

    let json = serde_json::to_string_pretty(&manifest)?;

    let _: serde_json::Value = serde_json::from_str(&json)?;

    println!("Manifest JSON ({} bytes):", json.len());
    println!("{}", &json[..json.len().min(500)]);

    Ok(())
}

#[test]
fn test_progress_is_reported_once_per_image() -> Result<()> {
    use keeper_vision::pipeline::CullProgress;
    use std::sync::Mutex;

    let models_dir = match get_models_dir() {
        Some(p) => p,
        None => return Ok(()),
    };
    let names = [
        "face-photo.JPG",
        "no-face-photo.JPG",
        "sharp-portrait.JPG",
        "soft-portrait.JPG",
    ];
    let mut images = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let Some(data) = load_test_jpeg(name) else {
            eprintln!("SKIPPED: {name} not found");
            return Ok(());
        };
        images.push(make_record(i as u64 + 1, name, None, data));
    }

    // Several workers, so reports arrive from different threads.
    let pipeline = CullPipeline::with_workers(&models_dir, CullConfig::default(), 3)?;
    let seen: Mutex<Vec<CullProgress>> = Mutex::new(Vec::new());
    let manifest = pipeline.run_with_progress(&images, &|p| seen.lock().unwrap().push(p))?;

    let mut seen = seen.into_inner().unwrap();
    assert_eq!(seen.len(), images.len(), "one report per image");
    assert!(seen.iter().all(|p| p.total == images.len()));
    seen.sort_by_key(|p| p.analysed);
    let counts: Vec<usize> = seen.iter().map(|p| p.analysed).collect();
    assert_eq!(
        counts,
        (1..=images.len()).collect::<Vec<_>>(),
        "counts 1..=total"
    );
    assert_eq!(manifest.total_images, images.len());
    Ok(())
}
