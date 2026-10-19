use anyhow::{Context, Result};
use image::DynamicImage;
use image_hasher::{HashAlg, HasherConfig};
use tracing::warn;

pub use image_hasher::ImageHash;

/// Long edge (px) images are shrunk to before hashing. The hash itself is
/// only 8×8, and hashing a full 24 MP preview took ~1.8 s (the hasher's
/// Lanczos resize); from this size it takes a few milliseconds.
const HASH_INPUT_LONG_EDGE: u32 = 640;

pub fn compute_phash(jpeg_bytes: &[u8]) -> Result<ImageHash> {
    let img = image::load_from_memory(jpeg_bytes)
        .context("Failed to decode JPEG for perceptual hashing")?;

    Ok(compute_phash_image(&img))
}

/// Hashes an already-decoded image. Large images are shrunk first with a
/// fast thumbnail filter; pass an image ≤ 640 px to skip that step.
pub fn compute_phash_image(img: &DynamicImage) -> ImageHash {
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(8, 8)
        .to_hasher();

    if img.width().max(img.height()) > HASH_INPUT_LONG_EDGE {
        hasher.hash_image(&img.thumbnail(HASH_INPUT_LONG_EDGE, HASH_INPUT_LONG_EDGE))
    } else {
        hasher.hash_image(img)
    }
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
