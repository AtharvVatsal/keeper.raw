use anyhow::{Context, Result};
use keeper_core::types::ImageRecord;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, warn};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn extract_preview(raw_path: &Path) -> Result<ImageRecord> {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let file_name = raw_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let preview_data = {
        let jpg_from_raw = Command::new("exiftool")
            .args(["-b", "-JpgFromRaw"])
            .arg(raw_path)
            .output()
            .context("Failed to run exiftool. Is it installed and in your PATH?")?;

        if !jpg_from_raw.stdout.is_empty() {
            jpg_from_raw.stdout
        } else {
            let preview = Command::new("exiftool")
                .args(["-b", "-PreviewImage"])
                .arg(raw_path)
                .output()?;

            if !preview.stdout.is_empty() {
                preview.stdout
            } else {
                let other = Command::new("exiftool")
                    .args(["-b", "-OtherImage"])
                    .arg(raw_path)
                    .output()?;
                other.stdout
            }
        }
    };

    if preview_data.is_empty() {
        anyhow::bail!("No embedded preview found in {}", file_name);
    }

    let (preview_width, preview_height) = {
        use image::ImageReader;
        use std::io::Cursor;

        ImageReader::new(Cursor::new(&preview_data))
            .with_guessed_format()
            .ok()
            .and_then(|reader| reader.into_dimensions().ok())
            .unwrap_or((0, 0))
    };

    let exif_output = Command::new("exiftool")
        .args([
            "-j",
            "-DateTimeOriginal",
            "-Make",
            "-Model",
            "-FocalLength",
            "-FNumber",
            "-ISO",
        ])
        .arg(raw_path)
        .output()?;

    let exif_json: serde_json::Value =
        serde_json::from_slice(&exif_output.stdout).unwrap_or_default();

    let exif = exif_json
        .as_array()
        .and_then(|arr| arr.first())
        .cloned()
        .unwrap_or_default();

    let timestamp = exif
        .get("DateTimeOriginal")
        .and_then(|v| v.as_str())
        .and_then(parse_exif_datetime);

    let camera_make = exif
        .get("Make")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let camera_model = exif
        .get("Model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let focal_length_mm = exif
        .get("FocalLength")
        .and_then(|v| v.as_str())
        .and_then(|s| s.replace(" mm", "").parse::<f32>().ok());

    let aperture = exif
        .get("FNumber")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    let iso = exif.get("ISO").and_then(|v| v.as_u64()).map(|v| v as u32);

    info!(
        "Extracted preview from {} ({}x{}, {} bytes)",
        file_name,
        preview_width,
        preview_height,
        preview_data.len()
    );

    Ok(ImageRecord {
        id,
        file_path: raw_path.to_path_buf(),
        file_name,
        timestamp,
        camera_make,
        camera_model,
        focal_length_mm,
        aperture,
        iso,
        preview_width,
        preview_height,
        preview_data,
    })
}

pub fn extract_all(paths: &[PathBuf]) -> Result<Vec<ImageRecord>> {
    let mut records = Vec::new();

    for (i, path) in paths.iter().enumerate() {
        match extract_preview(path) {
            Ok(record) => records.push(record),
            Err(e) => {
                warn!(
                    "Skipping {} ({}): {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    i + 1,
                    e
                );
            }
        }
    }

    info!(
        "Successfully extracted {}/{} previews",
        records.len(),
        paths.len()
    );
    Ok(records)
}

fn parse_exif_datetime(s: &str) -> Option<i64> {
    let parts: Vec<&str> = s.split([':', ' ']).collect();
    if parts.len() < 6 {
        return None;
    }

    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    let hour: u32 = parts[3].parse().ok()?;
    let min: u32 = parts[4].parse().ok()?;
    let sec: u32 = parts[5].parse().ok()?;

    let days_since_epoch = days_from_date(year, month, day)?;
    let total_seconds =
        (days_since_epoch as i64) * 86400 + (hour as i64) * 3600 + (min as i64) * 60 + (sec as i64);

    Some(total_seconds)
}

fn days_from_date(year: i32, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 { month + 9 } else { month - 3 } as i64;
    let d = day as i64;
    Some(365 * y + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + d - 719469)
}
