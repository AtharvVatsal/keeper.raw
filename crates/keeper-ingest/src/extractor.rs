use anyhow::{anyhow, Context, Result};
use keeper_core::types::ImageRecord;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// =============================================================================
// Batch extraction with ExifTool
//
// Starting exiftool costs ~0.5 s on Windows (the standalone .exe unpacks
// itself on every launch), and ingest used to launch it twice per file
// (~1.05 s/file). Now the files are split into a few chunks; for each chunk
// ONE exiftool call writes every embedded preview into a temp folder and ONE
// call returns all metadata as JSON. Chunks run in parallel.
// =============================================================================

/// Upper bound on exiftool processes running at once. Measured on 3,068
/// NEFs (16-core PC): 4 → 113 s, 6 → 82 s, 8 → 66 s, 12 → 64 s.
const MAX_PARALLEL_EXIFTOOL: usize = 8;

/// Embedded previews to try, best first. Each later tag is only asked for
/// the files the earlier ones didn't cover.
const PREVIEW_TAGS: [&str; 3] = ["JpgFromRaw", "PreviewImage", "OtherImage"];

const METADATA_ARGS: [&str; 8] = [
    "-j",
    "-DateTimeOriginal",
    "-Make",
    "-Model",
    "-FocalLength",
    "-FNumber",
    "-ISO",
    // `#` = numeric value (1–8) instead of text like "Rotate 270 CW".
    "-Orientation#",
];

/// Extracts the embedded preview and metadata of a single RAW file.
pub fn extract_preview(raw_path: &Path) -> Result<ImageRecord> {
    extract_all(&[raw_path.to_path_buf()])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("No embedded preview found in {}", raw_path.display()))
}

/// Extracts previews and metadata for all `paths`, in order, keeping each
/// preview in memory (`ImageRecord::preview_data`). Files without an
/// embedded preview are skipped with a warning. Fails only if exiftool can't
/// be run at all.
pub fn extract_all(paths: &[PathBuf]) -> Result<Vec<ImageRecord>> {
    extract_all_with(paths, PreviewStore::Memory)
}

/// Same as [`extract_all`], but each preview is moved into `preview_dir` as
/// `<id>.jpg` and referenced by `ImageRecord::preview_file` instead of being
/// held in memory. Previews are never read into RAM during ingest.
pub fn extract_all_to_dir(paths: &[PathBuf], preview_dir: &Path) -> Result<Vec<ImageRecord>> {
    std::fs::create_dir_all(preview_dir)
        .with_context(|| format!("Failed to create {}", preview_dir.display()))?;
    extract_all_with(paths, PreviewStore::Directory(preview_dir))
}

/// Where extracted previews end up.
#[derive(Clone, Copy)]
enum PreviewStore<'a> {
    Memory,
    Directory(&'a Path),
}

fn extract_all_with(paths: &[PathBuf], store: PreviewStore) -> Result<Vec<ImageRecord>> {
    if paths.is_empty() {
        return Ok(vec![]);
    }

    let work_dir = TempDir::new()?;
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(MAX_PARALLEL_EXIFTOOL)
        .min(paths.len());
    let chunk_size = paths.len().div_ceil(workers);

    // ---- Run the chunks in parallel ------------------------------------------
    let results: Vec<Result<ChunkOutput>> = std::thread::scope(|scope| {
        let handles: Vec<_> = paths
            .chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let dir = work_dir.path().join(format!("chunk{i}"));
                scope.spawn(move || extract_chunk(chunk, &dir))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow!("exiftool worker thread panicked")))
            })
            .collect()
    });

    // ---- Assemble records in input order --------------------------------------
    let mut records = Vec::with_capacity(paths.len());
    for (chunk, result) in paths.chunks(chunk_size).zip(results) {
        let output = result?;
        for ((path, written), exif) in chunk.iter().zip(output.previews).zip(output.metadata) {
            let Some(written) = written else {
                warn!("Skipping {}: no embedded preview found", path.display());
                continue;
            };
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let preview = match store {
                PreviewStore::Memory => Preview::Bytes(
                    std::fs::read(&written)
                        .with_context(|| format!("Failed to read {}", written.display()))?,
                ),
                PreviewStore::Directory(dir) => {
                    let dest = dir.join(format!("{id}.jpg"));
                    move_file(&written, &dest)?;
                    Preview::File(dest)
                }
            };
            records.push(build_record(id, path, preview, &exif));
        }
    }

    info!(
        "Successfully extracted {}/{} previews ({} exiftool worker(s))",
        records.len(),
        paths.len(),
        workers
    );
    Ok(records)
}

/// What one exiftool worker produced, one entry per input path.
struct ChunkOutput {
    /// The preview file exiftool wrote for each input (in the temp folder).
    previews: Vec<Option<PathBuf>>,
    metadata: Vec<serde_json::Value>,
}

fn extract_chunk(paths: &[PathBuf], dir: &Path) -> Result<ChunkOutput> {
    std::fs::create_dir_all(dir)?;

    // ---- Previews: one call per tag, only for files still missing one -------
    let mut previews: Vec<Option<PathBuf>> = vec![None; paths.len()];
    for tag in PREVIEW_TAGS {
        let missing: Vec<usize> = (0..paths.len())
            .filter(|&i| previews[i].is_none())
            .collect();
        if missing.is_empty() {
            break;
        }
        let tag_dir = dir.join(tag);
        std::fs::create_dir_all(&tag_dir)?;
        let argfile = write_argfile(
            dir,
            &format!("{tag}.args"),
            missing.iter().map(|&i| &paths[i]),
        )?;
        // `-w DIR/%f.%e.jpg` writes each file's preview to e.g. DIR/IMG_1.CR3.jpg.
        let pattern = format!("{}/%f.%e.jpg", tag_dir.to_string_lossy().replace('\\', "/"));
        run_exiftool(&["-b", &format!("-{tag}"), "-w", &pattern], &argfile)?;

        for &i in &missing {
            let file_name = paths[i].file_name().unwrap_or_default().to_string_lossy();
            let written = tag_dir.join(format!("{file_name}.jpg"));
            let non_empty = std::fs::metadata(&written)
                .map(|m| m.len() > 0)
                .unwrap_or(false);
            if non_empty {
                previews[i] = Some(written);
            }
        }
    }

    // ---- Metadata: one JSON call for the whole chunk -------------------------
    let argfile = write_argfile(dir, "metadata.args", paths.iter())?;
    let json = run_exiftool(&METADATA_ARGS, &argfile)?;
    let entries: Vec<serde_json::Value> = serde_json::from_slice(&json).unwrap_or_default();

    // Match entries to paths by exiftool's `SourceFile` (separators normalised);
    // fall back to position, which exiftool preserves.
    let key = |s: &str| s.replace('\\', "/");
    let by_source: HashMap<String, &serde_json::Value> = entries
        .iter()
        .filter_map(|e| {
            e.get("SourceFile")
                .and_then(|s| s.as_str())
                .map(|s| (key(s), e))
        })
        .collect();
    let metadata = paths
        .iter()
        .enumerate()
        .map(|(i, p)| {
            by_source
                .get(&key(&p.to_string_lossy()))
                .copied()
                .or_else(|| entries.get(i))
                .cloned()
                .unwrap_or_default()
        })
        .collect();

    Ok(ChunkOutput { previews, metadata })
}

/// Writes one path per line (UTF-8, read with `-charset filename=utf8`).
fn write_argfile<'a>(
    dir: &Path,
    name: &str,
    paths: impl Iterator<Item = &'a PathBuf>,
) -> Result<PathBuf> {
    let file = dir.join(name);
    let mut text = String::new();
    for p in paths {
        text.push_str(&p.to_string_lossy());
        text.push('\n');
    }
    std::fs::write(&file, text).with_context(|| format!("Failed to write {}", file.display()))?;
    Ok(file)
}

/// Runs exiftool on the files listed in `argfile` and returns its stdout.
/// A non-zero exit (e.g. some files lacked the tag) is not an error.
fn run_exiftool(args: &[&str], argfile: &Path) -> Result<Vec<u8>> {
    let output = Command::new("exiftool")
        .args(["-q", "-q", "-charset", "filename=utf8"])
        .args(args)
        .arg("-@")
        .arg(argfile)
        .output()
        .context("Failed to run exiftool. Is it installed and in your PATH?")?;
    Ok(output.stdout)
}

/// A uniquely named temp folder, deleted when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("keeper-raw-ingest-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create {}", dir.display()))?;
        Ok(TempDir(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A preview either held in memory or stored as a file.
enum Preview {
    Bytes(Vec<u8>),
    File(PathBuf),
}

/// Moves `from` to `to`; falls back to copy + delete across volumes.
fn move_file(from: &Path, to: &Path) -> Result<()> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to)
        .with_context(|| format!("Failed to copy {} to {}", from.display(), to.display()))?;
    let _ = std::fs::remove_file(from);
    Ok(())
}

/// Builds an `ImageRecord` from a preview and exiftool's JSON for one file.
fn build_record(
    id: u64,
    raw_path: &Path,
    preview: Preview,
    exif: &serde_json::Value,
) -> ImageRecord {
    let file_name = raw_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Only the JPEG header is read to get the size, also for files on disk.
    let (preview_width, preview_height) = {
        use image::ImageReader;
        use std::io::Cursor;

        match &preview {
            Preview::Bytes(bytes) => ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .ok()
                .and_then(|reader| reader.into_dimensions().ok()),
            Preview::File(path) => ImageReader::open(path)
                .ok()
                .and_then(|reader| reader.with_guessed_format().ok())
                .and_then(|reader| reader.into_dimensions().ok()),
        }
        .unwrap_or((0, 0))
    };
    let (preview_data, preview_file) = match preview {
        Preview::Bytes(bytes) => (bytes, None),
        Preview::File(path) => (Vec::new(), Some(path)),
    };

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

    debug!(
        "Extracted preview from {} ({}x{})",
        file_name, preview_width, preview_height
    );

    ImageRecord {
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
        preview_file,
    }
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
            preview_file: None,
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
