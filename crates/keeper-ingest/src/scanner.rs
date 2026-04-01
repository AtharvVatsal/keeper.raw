use std::path::{Path, PathBuf};
use tracing::info;

use crate::RAW_EXTENSIONS;

pub fn find_raw_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("Path is not a directory: {:?}", dir);
    }

    let mut results = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if RAW_EXTENSIONS.contains(&ext_lower.as_str()) {
                results.push(path);
            }
        }
    }

    results.sort();

    info!("Found {} RAW files in {:?}", results.len(), dir);
    Ok(results)
}
