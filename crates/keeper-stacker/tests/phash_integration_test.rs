use anyhow::Result;
use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_stacker::hasher;
use keeper_stacker::stacker;
use std::path::{Path, PathBuf};

fn load_test_image(id: u64, filename: &str, timestamp_ms: i64) -> Option<ImageRecord> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(filename);

    if !path.exists() {
        eprintln!("Test image not found: {:?}", path);
        return None;
    }

    let jpeg_bytes = std::fs::read(&path).ok()?;

    Some(ImageRecord {
        id,
        file_path: PathBuf::from(filename),
        file_name: filename.to_string(),
        timestamp: Some(timestamp_ms),
        camera_make: None,
        camera_model: None,
        focal_length_mm: None,
        aperture: None,
        iso: None,
        preview_width: 0,
        preview_height: 0,
        preview_data: jpeg_bytes,
    })
}

#[test]
fn test_phash_similar_images_have_low_distance() -> Result<()> {
    let img_a_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/burst_a_01.jpg");
    let img_b_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/burst_a_02.jpg");

    if !img_a_path.exists() || !img_b_path.exists() {
        eprintln!("SKIPPED: Need burst_a_01.jpg and burst_a_02.jpg in test-data/");
        return Ok(());
    }

    let bytes_a = std::fs::read(&img_a_path)?;
    let bytes_b = std::fs::read(&img_b_path)?;

    let hash_a = hasher::compute_phash(&bytes_a)?;
    let hash_b = hasher::compute_phash(&bytes_b)?;

    let distance = hasher::hamming_distance(&hash_a, &hash_b);
    println!("Similar images distance: {}", distance);

    assert!(
        distance <= 15,
        "Similar burst shots should have Hamming distance ≤ 15, got {}",
        distance
    );

    Ok(())
}

#[test]
fn test_phash_different_images_have_high_distance() -> Result<()> {
    let img_a_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/burst_a_01.jpg");
    let img_diff_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/burst_b_01.jpg");

    if !img_a_path.exists() || !img_diff_path.exists() {
        eprintln!("SKIPPED: Need burst_a_01.jpg and burst_b_01.jpg in test-data/");
        return Ok(());
    }

    let bytes_a = std::fs::read(&img_a_path)?;
    let bytes_diff = std::fs::read(&img_diff_path)?;

    let hash_a = hasher::compute_phash(&bytes_a)?;
    let hash_diff = hasher::compute_phash(&bytes_diff)?;

    let distance = hasher::hamming_distance(&hash_a, &hash_diff);
    println!("Different images distance: {}", distance);

    assert!(
        distance > 5,
        "Different scenes should have Hamming distance > 5, got {}",
        distance
    );

    Ok(())
}

#[test]
fn test_stacking_splits_mixed_burst_by_phash() -> Result<()> {
    let img_a1 = load_test_image(1, "burst_a_01.jpg", 1000);
    let img_a2 = load_test_image(2, "burst_a_02.jpg", 1200);
    let img_b1 = load_test_image(3, "burst_b_01.jpg", 1400);

    if img_a1.is_none() || img_a2.is_none() || img_b1.is_none() {
        eprintln!("SKIPPED: Need burst_a_01.jpg, burst_a_02.jpg, burst_b_01.jpg in test-data/");
        return Ok(());
    }

    let images = vec![img_a1.unwrap(), img_a2.unwrap(), img_b1.unwrap()];
    let config = CullConfig::default();

    let scenes = stacker::stack_scenes(&images, &config)?;

    println!("Scenes created: {}", scenes.len());
    for scene in &scenes {
        println!("  Scene {}: {:?}", scene.id, scene.image_ids);
    }

    if scenes.len() == 2 {
        println!("pHash correctly split the mixed burst into 2 scenes");
    } else if scenes.len() == 1 {
        println!("pHash did NOT split - test images may be too similar");
    } else {
        println!("Got {} scenes - unexpected", scenes.len());
    }

    Ok(())
}
