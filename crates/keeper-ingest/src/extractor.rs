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
            // `#` = numeric value (1–8) instead of text like "Rotate 270 CW".
            "-Orientation#",
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

    // Portrait shots are stored sideways; the embedded preview is NOT rotated
    // and has no orientation tag of its own, so the RAW's tag is the only
    // source. Missing/invalid → 1 (upright).
    let orientation = exif
        .get("Orientation")
        .and_then(|v| v.as_u64())
        .filter(|v| (1..=8).contains(v))
        .map(|v| v as u8)
        .unwrap_or(1);

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
        orientation,
        preview_data,
    })
}

// =============================================================================
// Orientation tag for display
//
// The webview shows the temp preview JPEG with an <img>, and browsers rotate
// images according to their EXIF Orientation tag. Embedded RAW previews have
// no EXIF at all, so we insert a minimal EXIF block holding only that tag.
// =============================================================================

/// Returns `jpeg` with an EXIF APP1 segment containing just `Orientation`
/// inserted right after the start-of-image marker. Returns the bytes unchanged
/// if `orientation` is 1 (upright) or not a valid EXIF value, or if `jpeg`
/// doesn't start with a JPEG SOI marker.
pub fn with_orientation_tag(jpeg: &[u8], orientation: u8) -> Vec<u8> {
    if orientation == 1 || !(1..=8).contains(&orientation) || !jpeg.starts_with(&[0xFF, 0xD8]) {
        return jpeg.to_vec();
    }

    #[rustfmt::skip]
    let app1: [u8; 36] = [
        0xFF, 0xE1, 0x00, 0x22,             // APP1 marker, segment length 34
        b'E', b'x', b'i', b'f', 0x00, 0x00, // "Exif\0\0"
        b'M', b'M', 0x00, 0x2A,             // TIFF header: big-endian, magic 42
        0x00, 0x00, 0x00, 0x08,             //   offset of IFD0
        0x00, 0x01,                         // IFD0: 1 entry
        0x01, 0x12,                         //   tag 0x0112 = Orientation
        0x00, 0x03,                         //   type SHORT
        0x00, 0x00, 0x00, 0x01,             //   count 1
        0x00, orientation, 0x00, 0x00,      //   value (left-aligned)
        0x00, 0x00, 0x00, 0x00,             // no next IFD
    ];

    let mut out = Vec::with_capacity(jpeg.len() + app1.len());
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpeg[2..]);
    out
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

/// Parses an EXIF `DateTimeOriginal` string ("YYYY:MM:DD HH:MM:SS") into
/// **milliseconds** since the Unix epoch — the unit `ImageRecord::timestamp`
/// and the stacker expect. The camera's local time is treated as UTC; only
/// the gaps between shots matter for stacking.
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
        days_since_epoch * 86400 + (hour as i64) * 3600 + (min as i64) * 60 + (sec as i64);

    Some(total_seconds * 1000)
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

#[cfg(test)]
mod tests {
    use super::*;
    use keeper_core::config::CullConfig;

    fn record_at(id: u64, exif_datetime: &str) -> ImageRecord {
        ImageRecord {
            id,
            file_path: PathBuf::from(format!("IMG_{id}.CR3")),
            file_name: format!("IMG_{id}.CR3"),
            timestamp: parse_exif_datetime(exif_datetime),
            camera_make: None,
            camera_model: None,
            focal_length_mm: None,
            aperture: None,
            iso: None,
            preview_width: 0,
            preview_height: 0,
            orientation: 1,
            preview_data: vec![],
        }
    }

    fn small_jpeg() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(40, 20, image::Rgb([200, 100, 50]));
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut bytes)
            .encode_image(&img)
            .unwrap();
        bytes
    }

    fn read_orientation(jpeg: &[u8]) -> image::metadata::Orientation {
        use image::ImageDecoder;
        let mut decoder =
            image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg)).unwrap();
        decoder.orientation().unwrap()
    }

    #[test]
    fn test_orientation_tag_is_readable_and_image_still_decodes() {
        use image::metadata::Orientation;
        let tagged = with_orientation_tag(&small_jpeg(), 6);
        assert_eq!(read_orientation(&tagged), Orientation::Rotate90);
        assert_eq!(
            read_orientation(&with_orientation_tag(&small_jpeg(), 8)),
            Orientation::Rotate270
        );

        let decoded = image::load_from_memory(&tagged).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (40, 20));
    }

    #[test]
    fn test_orientation_tag_skipped_when_upright_or_invalid() {
        let jpeg = small_jpeg();
        assert_eq!(with_orientation_tag(&jpeg, 1), jpeg);
        assert_eq!(with_orientation_tag(&jpeg, 0), jpeg);
        assert_eq!(with_orientation_tag(&jpeg, 9), jpeg);
        assert_eq!(with_orientation_tag(b"not a jpeg", 6), b"not a jpeg");
    }

    #[test]
    fn test_parse_exif_datetime_returns_unix_millis() {
        // 2024-06-15 14:30:00 UTC = 1_718_461_800 seconds since the Unix epoch.
        assert_eq!(
            parse_exif_datetime("2024:06:15 14:30:00"),
            Some(1_718_461_800_000)
        );
    }

    #[test]
    fn test_parse_exif_datetime_rejects_garbage() {
        assert_eq!(parse_exif_datetime(""), None);
        assert_eq!(parse_exif_datetime("2024:13:01 00:00:00"), None);
        assert_eq!(parse_exif_datetime("not a date"), None);
    }

    // Regression: ingest used to return seconds while the stacker expects
    // milliseconds, so the default 3 s burst threshold acted like 50 minutes.
    #[test]
    fn test_parsed_timestamps_respect_burst_threshold() {
        let config = CullConfig::default(); // burst_threshold_secs = 3.0

        let images = vec![
            record_at(1, "2024:06:15 14:30:00"),
            record_at(2, "2024:06:15 14:30:02"), // 2 s later → same burst
            record_at(3, "2024:06:15 14:30:07"), // 5 s later → new scene
            record_at(4, "2024:06:15 14:40:07"), // 10 min later → new scene
        ];

        let scenes = keeper_stacker::stacker::stack_scenes(&images, &config).unwrap();
        let groups: Vec<Vec<u64>> = scenes.into_iter().map(|s| s.image_ids).collect();

        assert_eq!(groups, vec![vec![1, 2], vec![3], vec![4]]);
    }
}
