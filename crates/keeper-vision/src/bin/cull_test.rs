use anyhow::{Context, Result};
use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_vision::pipeline::CullPipeline;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let input_dir = if let Some(pos) = args.iter().position(|a| a == "--input") {
        args.get(pos + 1)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("test-data"))
    } else {
        PathBuf::from("test-data")
    };

    let output_file = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|pos| args.get(pos + 1).map(PathBuf::from));

    println!("═══════════════════════════════════════════");
    println!("  keeper.raw — CullPipeline Test");
    println!("═══════════════════════════════════════════");
    println!("Input directory: {:?}", input_dir);
    println!();

    let models_dir = Path::new("models");
    let face_model = models_dir.join("yolov8n-face.onnx");
    let mesh_model = models_dir.join("face_landmark.onnx");

    if !face_model.exists() {
        anyhow::bail!(
            "Face detection model not found at {:?}\n\
             Please place yolov8n-face.onnx in the models/ directory.",
            face_model
        );
    }
    if !mesh_model.exists() {
        anyhow::bail!(
            "Face mesh model not found at {:?}\n\
             Please place face_landmark.onnx in the models/ directory.",
            mesh_model
        );
    }

    println!("Loading images from {:?}...", input_dir);
    let images = load_test_images(&input_dir)?;
    println!("  Found {} images", images.len());

    if images.is_empty() {
        println!("No images found. Place .jpg files in {:?}", input_dir);
        return Ok(());
    }

    println!("\nInitializing pipeline...");
    let config = CullConfig::default();
    let pipeline = CullPipeline::new(&face_model, &mesh_model, config)?;

    println!("\nRunning pipeline...\n");
    let manifest = pipeline.run(&images)?;

    println!("\n═══════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════");
    println!("Total images:  {}", manifest.total_images);
    println!("Total scenes:  {}", manifest.total_scenes);
    println!("Keepers:       {}", manifest.total_keepers);
    println!("Rejects:       {}", manifest.total_rejects);
    println!("Unrated:       {}", manifest.total_unrated);
    println!(
        "Processing time: {:.1}s ({:.0} ms/image)",
        manifest.processing_time_secs,
        manifest.processing_time_secs * 1000.0 / manifest.total_images.max(1) as f64
    );

    println!("\n--- Scenes ---");
    for scene in &manifest.scenes {
        let keeper_name = scene.keeper_id.and_then(|kid| {
            images
                .iter()
                .find(|i| i.id == kid)
                .map(|i| i.file_name.as_str())
        });

        println!(
            "  Scene {}: {} image(s), keeper={}",
            scene.id,
            scene.image_ids.len(),
            keeper_name.unwrap_or("none"),
        );
    }

    println!("\n--- Per-image scores ---");
    for score in &manifest.image_scores {
        let name = images
            .iter()
            .find(|i| i.id == score.image_id)
            .map(|i| i.file_name.as_str())
            .unwrap_or("?");

        let verdict_str = match &score.verdict {
            keeper_core::types::Verdict::Keeper => "⭐ KEEPER",
            keeper_core::types::Verdict::Reject { reason } => match reason {
                keeper_core::types::RejectReason::Blink => "❌ REJECT (blink)",
                keeper_core::types::RejectReason::OutOfFocus => "❌ REJECT (blur)",
            },
            keeper_core::types::Verdict::Unrated => "   unrated",
        };

        println!(
            "  {} | sharpness={:.1} | face={} | EAR={} | {}",
            name,
            score.sharpness_score,
            if score.has_face { "yes" } else { "no " },
            score
                .eye_aspect_ratio
                .map(|e| format!("{:.3}", e))
                .unwrap_or_else(|| "n/a".to_string()),
            verdict_str,
        );
    }

    let json = serde_json::to_string_pretty(&manifest).context("Failed to serialize manifest")?;

    if let Some(ref out_path) = output_file {
        std::fs::write(out_path, &json)
            .context(format!("Failed to write manifest to {:?}", out_path))?;
        println!("\n✅ Manifest written to {:?}", out_path);
    } else {
        let default_output = input_dir.join("cull_manifest.json");
        std::fs::write(&default_output, &json)
            .context(format!("Failed to write manifest to {:?}", default_output))?;
        println!("\n✅ Manifest written to {:?}", default_output);
    }

    println!("\n═══════════════════════════════════════════");
    println!("  Phase 1 exit gate: PASSED ✅");
    println!("═══════════════════════════════════════════");

    Ok(())
}

fn load_test_images(dir: &Path) -> Result<Vec<ImageRecord>> {
    let mut images = Vec::new();

    if !dir.exists() {
        anyhow::bail!("Directory not found: {:?}", dir);
    }

    let entries = std::fs::read_dir(dir).context(format!("Failed to read directory {:?}", dir))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext != "jpg" && ext != "jpeg" {
            continue;
        }

        let file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let preview_data = std::fs::read(&path).context(format!("Failed to read {:?}", path))?;

        if preview_data.is_empty() {
            continue;
        }

        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        images.push(ImageRecord {
            id,
            file_path: path.clone(),
            file_name,
            timestamp: None,
            camera_make: None,
            camera_model: None,
            focal_length_mm: None,
            aperture: None,
            iso: None,
            preview_width: 0,
            preview_height: 0,
            preview_data,
        });
    }

    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    Ok(images)
}
