use anyhow::{Context, Result};
use image_hasher::{HashAlg, HasherConfig, ImageHash};
use tracing::warn;

pub fn compute_phash(jpeg_bytes: &[u8]) -> Result<ImageHash> {
    let img = image::load_from_memory(jpeg_bytes)
        .context("Failed to decode JPEG for perceptual hashing")?;

    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(8, 8)
        .to_hasher();

    let hash = hasher.hash_image(&img);
    Ok(hash)
}

pub fn hamming_distance(hash_a: &ImageHash, hash_b: &ImageHash) -> u32 {
    hash_a.dist(hash_b)
}

pub fn try_compute_phash(jpeg_bytes: &[u8], file_name: &str) -> Option<ImageHash> {
    match compute_phash(jpeg_bytes) {
        Ok(hash) => Some(hash),
        Err(e) => {
            warn!("Could not compute pHash for '{}': {}", file_name, e);
            None
        }
    }
}
