pub mod extractor;
pub mod scanner;

use keeper_core::types::ImageRecord;
use std::path::Path;

pub const RAW_EXTENSIONS: &[&str] = &["cr2", "cr3", "nef", "arw", "raf", "dng", "orf", "rw2"];

pub fn scan_directory(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    scanner::find_raw_files(dir)
}

pub fn ingest_directory(dir: &Path) -> anyhow::Result<Vec<ImageRecord>> {
    let paths = scanner::find_raw_files(dir)?;
    let records = extractor::extract_all(&paths)?;
    Ok(records)
}
