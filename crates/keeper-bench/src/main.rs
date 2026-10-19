//! keeper-bench — pipeline benchmarking harness for keeper.raw
//!
//! Loads a folder of images, runs the cull pipeline N times, and reports
//! per-stage timings and throughput. Lightweight: uses `std::time::Instant`
//! only — no statistical-benchmark framework.
//!
//! Usage:
//!   keeper-bench [--input <dir>] [--runs <n>] [--warmup]
//!                [--models <dir>] [--raw]
//!
//!   --input <dir>   Folder of images to benchmark (default: test-data)
//!   --runs <n>      Number of timed pipeline runs (default: 3)
//!   --warmup        Run one untimed warmup pass before timing
//!   --models <dir>  Directory holding the ONNX models (default: models)
//!   --raw           Treat input as RAW files and ingest via ExifTool.
//!                   Default treats input files as JPEG previews directly.

use anyhow::{Context, Result};
use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_vision::pipeline::CullPipeline;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct Args {
    input_dir: PathBuf,
    models_dir: PathBuf,
    runs: usize,
    warmup: bool,
    raw: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();

    let value_after = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|pos| argv.get(pos + 1).cloned())
    };

    let runs = value_after("--runs")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(3);

    Args {
        input_dir: value_after("--input")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("test-data")),
        models_dir: value_after("--models")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models")),
        runs,
        warmup: argv.iter().any(|a| a == "--warmup"),
        raw: argv.iter().any(|a| a == "--raw"),
    }
}

fn main() -> Result<()> {
    let args = parse_args();

    println!("═══════════════════════════════════════════");
    println!("  keeper.raw — pipeline benchmark");
    println!("═══════════════════════════════════════════");
    println!("Input:   {:?}", args.input_dir);
    println!("Models:  {:?}", args.models_dir);
    println!(
        "Runs:    {}{}",
        args.runs,
        if args.warmup { " (+1 warmup)" } else { "" }
    );
    println!(
        "Mode:    {}",
        if args.raw {
            "RAW (ExifTool)"
        } else {
            "JPEG previews"
        }
    );
    println!();

    let face_model = args.models_dir.join("yolov8n-face.onnx");
    let mesh_model = args.models_dir.join("face_landmark.onnx");

    if !face_model.exists() {
        anyhow::bail!(
            "Face detection model not found at {:?}\n\
             Place yolov8n-face.onnx in {:?} or pass --models <dir>.",
            face_model,
            args.models_dir
        );
    }
    if !mesh_model.exists() {
        anyhow::bail!(
            "Face mesh model not found at {:?}\n\
             Place face_landmark.onnx in {:?} or pass --models <dir>.",
            mesh_model,
            args.models_dir
        );
    }

    // ── Load images ────────────────────────────────────────────────────────
    let load_start = Instant::now();
    let images = if args.raw {
        keeper_ingest::ingest_directory(&args.input_dir)
            .context("Failed to ingest RAW files. Is ExifTool installed and on PATH?")?
    } else {
        load_jpeg_images(&args.input_dir)?
    };
    let load_elapsed = load_start.elapsed();

    if images.is_empty() {
        anyhow::bail!(
            "No images found in {:?}. Place images there or pass --input <dir>.",
            args.input_dir
        );
    }
    println!(
        "Loaded {} images in {} ({})",
        images.len(),
        fmt_duration(load_elapsed),
        if args.raw {
            "ingest + preview extraction"
        } else {
            "raw file read"
        }
    );

    let config = CullConfig::default();

    // ── Model load (pipeline init) ─────────────────────────────────────────
    let init_start = Instant::now();
    let pipeline = CullPipeline::new(&face_model, &mesh_model, config.clone())
        .context("Failed to initialize CullPipeline")?;
    let init_elapsed = init_start.elapsed();
    println!("Model load (pipeline init): {}", fmt_duration(init_elapsed));

    // ── Stacking in isolation (cheap, CPU-only) ────────────────────────────
    let stack_start = Instant::now();
    let scenes =
        keeper_stacker::stacker::stack_scenes(&images, &config).context("Scene stacking failed")?;
    let stack_elapsed = stack_start.elapsed();
    println!(
        "Stacking (standalone):      {}  → {} scenes",
        fmt_duration(stack_elapsed),
        scenes.len()
    );
    println!();

    // ── Timed pipeline runs ────────────────────────────────────────────────
    if args.warmup {
        print!("Warmup run... ");
        let _ = pipeline.run(&images).context("Warmup run failed")?;
        println!("done");
    }

    let mut wall_times: Vec<Duration> = Vec::with_capacity(args.runs);
    let mut reported_secs: Vec<f64> = Vec::with_capacity(args.runs);
    let mut last_manifest = None;

    for run in 1..=args.runs {
        let start = Instant::now();
        let manifest = pipeline
            .run(&images)
            .with_context(|| format!("Pipeline run {} failed", run))?;
        let elapsed = start.elapsed();

        println!(
            "Run {}/{}: {}  ({:.1} ms/image)",
            run,
            args.runs,
            fmt_duration(elapsed),
            elapsed.as_secs_f64() * 1000.0 / images.len() as f64
        );

        wall_times.push(elapsed);
        reported_secs.push(manifest.processing_time_secs);
        last_manifest = Some(manifest);
    }
    println!();

    // ── Report ─────────────────────────────────────────────────────────────
    let manifest = last_manifest.expect("at least one run completed");
    let (min, avg, max) = stats(&wall_times);
    let avg_reported = reported_secs.iter().sum::<f64>() / reported_secs.len() as f64;
    let per_image_ms = avg.as_secs_f64() * 1000.0 / images.len() as f64;
    // Full run includes stacking internally; isolate the vision share.
    let vision_share = avg.as_secs_f64() - stack_elapsed.as_secs_f64();

    println!("═══════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════");
    println!("Images:            {}", images.len());
    println!("Scenes:            {}", manifest.total_scenes);
    println!(
        "Verdicts:          {} keepers / {} rejects / {} unrated",
        manifest.total_keepers, manifest.total_rejects, manifest.total_unrated
    );
    println!();
    println!("Model load:        {}", fmt_duration(init_elapsed));
    println!("Image load:        {}", fmt_duration(load_elapsed));
    println!("Stacking:          {}", fmt_duration(stack_elapsed));
    println!(
        "Vision (≈run-stack): {}",
        fmt_duration(Duration::from_secs_f64(vision_share.max(0.0)))
    );
    println!();
    println!("Pipeline wall time ({} runs):", wall_times.len());
    println!("  min: {}", fmt_duration(min));
    println!("  avg: {}", fmt_duration(avg));
    println!("  max: {}", fmt_duration(max));
    println!("  reported (manifest avg): {:.3}s", avg_reported);
    println!();
    println!(
        "Throughput:        {:.1} ms/image  ({:.1} images/s)",
        per_image_ms,
        1000.0 / per_image_ms
    );
    println!("═══════════════════════════════════════════");

    Ok(())
}

/// Load image files directly as JPEG previews (no RAW decode / ExifTool).
/// Timestamps are left empty, so stacking yields singleton scenes — this
/// isolates the vision pipeline as the dominant cost for benchmarking.
fn load_jpeg_images(dir: &Path) -> Result<Vec<ImageRecord>> {
    if !dir.exists() {
        anyhow::bail!("Directory not found: {:?}", dir);
    }

    let mut images = Vec::new();

    for entry in std::fs::read_dir(dir).with_context(|| format!("Failed to read {:?}", dir))? {
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

        let preview_data =
            std::fs::read(&path).with_context(|| format!("Failed to read {:?}", path))?;
        if preview_data.is_empty() {
            continue;
        }

        let file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
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

fn stats(times: &[Duration]) -> (Duration, Duration, Duration) {
    let min = times.iter().min().copied().unwrap_or_default();
    let max = times.iter().max().copied().unwrap_or_default();
    let sum: Duration = times.iter().sum();
    let avg = if times.is_empty() {
        Duration::default()
    } else {
        sum / times.len() as u32
    };
    (min, avg, max)
}

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.0}ms", secs * 1000.0)
    }
}
