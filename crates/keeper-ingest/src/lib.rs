pub mod extractor;
pub use extractor::with_orientation_tag;
pub mod scanner;

use anyhow::Context;
use keeper_core::types::ImageRecord;
use std::path::Path;

pub const RAW_EXTENSIONS: &[&str] = &["cr2", "cr3", "nef", "arw", "raf", "dng", "orf", "rw2"];

pub fn scan_directory(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    scanner::find_raw_files(dir)
}

/// Ingests every RAW file in `dir`, keeping previews in memory.
pub fn ingest_directory(dir: &Path) -> anyhow::Result<Vec<ImageRecord>> {
    let paths = scanner::find_raw_files(dir)?;
    let records = extractor::extract_all(&paths)?;
    Ok(records)
}

/// Ingests every RAW file in `dir`, storing each preview as
/// `preview_dir/<id>.jpg` (referenced by `ImageRecord::preview_file`)
/// instead of holding it in memory.
pub fn ingest_directory_to(dir: &Path, preview_dir: &Path) -> anyhow::Result<Vec<ImageRecord>> {
    let paths = scanner::find_raw_files(dir)?;
    extractor::extract_all_to_dir(&paths, preview_dir)
}

/// Adds the EXIF orientation tag to each rotated record's preview file, in
/// place, so a web view displays it upright (see [`with_orientation_tag`]).
/// Decoding the tagged file gives exactly the same pixels.
pub fn tag_preview_files(records: &[ImageRecord]) -> anyhow::Result<()> {
    for record in records.iter().filter(|r| r.orientation != 1) {
        if let Some(path) = &record.preview_file {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            std::fs::write(path, with_orientation_tag(&bytes, record.orientation))
                .with_context(|| format!("Failed to write {}", path.display()))?;
        }
    }
    Ok(())
}
