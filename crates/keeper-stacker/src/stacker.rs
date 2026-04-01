use anyhow::Result;
use image_hasher::ImageHash;
use keeper_core::config::CullConfig;
use keeper_core::types::{ImageId, ImageRecord, Scene, SceneId};
use tracing::{debug, info};

use crate::hasher;

pub fn stack_scenes(images: &[ImageRecord], config: &CullConfig) -> Result<Vec<Scene>> {
    if images.is_empty() {
        info!("No images to stack.");
        return Ok(vec![]);
    }

    info!("Stacking {} images into scenes...", images.len());

    let time_clusters = cluster_by_timestamp(images, config.burst_threshold_secs);
    info!(
        "Timestamp clustering: {} clusters from {} images",
        time_clusters.len(),
        images.len()
    );

    let mut scenes: Vec<Scene> = Vec::new();
    let mut next_scene_id: SceneId = 1;

    for cluster in &time_clusters {
        let cluster_records: Vec<&ImageRecord> = cluster
            .iter()
            .filter_map(|id| images.iter().find(|img| img.id == *id))
            .collect();

        let sub_groups = split_by_phash(&cluster_records, config.phash_similarity_threshold);

        for group in sub_groups {
            scenes.push(Scene {
                id: next_scene_id,
                image_ids: group,
                keeper_id: None,
            });
            next_scene_id += 1;
        }
    }

    info!(
        "Scene stacking complete: {} scenes (from {} time clusters)",
        scenes.len(),
        time_clusters.len()
    );

    let singleton_count = scenes.iter().filter(|s| s.image_ids.len() == 1).count();
    let largest_scene = scenes.iter().map(|s| s.image_ids.len()).max().unwrap_or(0);
    let avg_scene_size = if scenes.is_empty() {
        0.0
    } else {
        images.len() as f64 / scenes.len() as f64
    };
    info!(
        "  Singletons: {}, Largest scene: {} images, Average: {:.1} images/scene",
        singleton_count, largest_scene, avg_scene_size
    );

    Ok(scenes)
}

fn cluster_by_timestamp(images: &[ImageRecord], threshold_secs: f64) -> Vec<Vec<ImageId>> {
    let mut with_timestamps: Vec<&ImageRecord> = images
        .iter()
        .filter(|img| img.timestamp.is_some())
        .collect();

    let without_timestamps: Vec<&ImageRecord> = images
        .iter()
        .filter(|img| img.timestamp.is_none())
        .collect();

    with_timestamps.sort_by_key(|img| img.timestamp.unwrap());

    let threshold_ms = (threshold_secs * 1000.0) as i64;
    let mut clusters: Vec<Vec<ImageId>> = Vec::new();

    if !with_timestamps.is_empty() {
        let mut current_cluster = vec![with_timestamps[0].id];

        for i in 1..with_timestamps.len() {
            let prev_ts = with_timestamps[i - 1].timestamp.unwrap();
            let curr_ts = with_timestamps[i].timestamp.unwrap();
            let gap_ms = curr_ts - prev_ts;

            if gap_ms <= threshold_ms {
                current_cluster.push(with_timestamps[i].id);
            } else {
                clusters.push(current_cluster);
                current_cluster = vec![with_timestamps[i].id];
            }
        }
        clusters.push(current_cluster);
    }

    for img in &without_timestamps {
        debug!(
            "Image '{}' has no timestamp — creating singleton scene",
            img.file_name
        );
        clusters.push(vec![img.id]);
    }

    clusters
}

fn split_by_phash(cluster_records: &[&ImageRecord], threshold: u32) -> Vec<Vec<ImageId>> {
    if cluster_records.len() <= 1 {
        return vec![cluster_records.iter().map(|r| r.id).collect()];
    }

    let hashes: Vec<Option<ImageHash>> = cluster_records
        .iter()
        .map(|record| {
            if record.preview_data.is_empty() {
                None
            } else {
                hasher::try_compute_phash(&record.preview_data, &record.file_name)
            }
        })
        .collect();

    let mut sub_groups: Vec<Vec<ImageId>> = Vec::new();
    let mut current_group = vec![cluster_records[0].id];
    let mut anchor_hash: Option<&ImageHash> = hashes[0].as_ref();

    for i in 1..cluster_records.len() {
        let should_split = match (anchor_hash, &hashes[i]) {
            (Some(anchor), Some(current)) => {
                let dist = hasher::hamming_distance(anchor, current);
                debug!(
                    "pHash distance: {} vs {} = {} (threshold: {})",
                    cluster_records[0].file_name, cluster_records[i].file_name, dist, threshold
                );
                dist > threshold
            }
            _ => false,
        };

        if should_split {
            sub_groups.push(current_group);
            current_group = vec![cluster_records[i].id];
            anchor_hash = hashes[i].as_ref();
        } else {
            current_group.push(cluster_records[i].id);
        }
    }
    sub_groups.push(current_group);

    sub_groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use keeper_core::types::ImageRecord;
    use std::path::PathBuf;

    fn make_image(id: ImageId, timestamp_ms: Option<i64>) -> ImageRecord {
        ImageRecord {
            id,
            file_path: PathBuf::from(format!("test_{}.cr3", id)),
            file_name: format!("test_{}.cr3", id),
            timestamp: timestamp_ms,
            camera_make: None,
            camera_model: None,
            focal_length_mm: None,
            aperture: None,
            iso: None,
            preview_width: 0,
            preview_height: 0,
            preview_data: vec![],
        }
    }

    #[test]
    fn test_empty_input() {
        let config = CullConfig::default();
        let scenes = stack_scenes(&[], &config).unwrap();
        assert!(scenes.is_empty());
    }

    #[test]
    fn test_single_image() {
        let config = CullConfig::default();
        let images = vec![make_image(1, Some(1000))];
        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].image_ids, vec![1]);
    }

    #[test]
    fn test_tight_burst_grouped_together() {
        let config = CullConfig::default();

        let images = vec![
            make_image(1, Some(1000)),
            make_image(2, Some(1200)),
            make_image(3, Some(1400)),
            make_image(4, Some(1600)),
            make_image(5, Some(1800)),
        ];

        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 1, "All 5 images should be in one scene");
        assert_eq!(scenes[0].image_ids.len(), 5);
    }

    #[test]
    fn test_two_separate_bursts() {
        let config = CullConfig::default();

        let images = vec![
            make_image(1, Some(0)),
            make_image(2, Some(200)),
            make_image(3, Some(400)),
            make_image(4, Some(5000)),
            make_image(5, Some(5200)),
            make_image(6, Some(5400)),
        ];

        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 2, "Should produce 2 separate scenes");
        assert_eq!(scenes[0].image_ids, vec![1, 2, 3]);
        assert_eq!(scenes[1].image_ids, vec![4, 5, 6]);
    }

    #[test]
    fn test_images_without_timestamps_become_singletons() {
        let config = CullConfig::default();

        let images = vec![
            make_image(1, Some(1000)),
            make_image(2, Some(1200)),
            make_image(3, None),
            make_image(4, None),
        ];

        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 3);
    }

    #[test]
    fn test_custom_threshold() {
        let config = CullConfig {
            burst_threshold_secs: 0.5,
            ..CullConfig::default()
        };

        let images = vec![
            make_image(1, Some(0)),
            make_image(2, Some(300)),
            make_image(3, Some(900)),
            make_image(4, Some(1100)),
        ];

        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0].image_ids, vec![1, 2]);
        assert_eq!(scenes[1].image_ids, vec![3, 4]);
    }

    #[test]
    fn test_unsorted_input_gets_sorted() {
        let images = vec![
            make_image(3, Some(400)),
            make_image(1, Some(0)),
            make_image(5, Some(5200)),
            make_image(2, Some(200)),
            make_image(4, Some(5000)),
        ];

        let config = CullConfig::default();
        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0].image_ids, vec![1, 2, 3]);
        assert_eq!(scenes[1].image_ids, vec![4, 5]);
    }

    #[test]
    fn test_all_singletons_when_large_gaps() {
        let config = CullConfig::default();

        let images = vec![
            make_image(1, Some(0)),
            make_image(2, Some(10_000)),
            make_image(3, Some(20_000)),
            make_image(4, Some(30_000)),
        ];

        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 4, "Each image should be its own scene");
        for scene in &scenes {
            assert_eq!(scene.image_ids.len(), 1);
        }
    }

    #[test]
    fn test_keeper_id_is_none_after_stacking() {
        let config = CullConfig::default();
        let images = vec![make_image(1, Some(0)), make_image(2, Some(200))];

        let scenes = stack_scenes(&images, &config).unwrap();
        for scene in &scenes {
            assert!(scene.keeper_id.is_none());
        }
    }

    #[test]
    fn test_long_continuous_burst() {
        let config = CullConfig::default();

        let images: Vec<ImageRecord> = (0..20)
            .map(|i| make_image(i + 1, Some(i as i64 * 100)))
            .collect();

        let scenes = stack_scenes(&images, &config).unwrap();
        assert_eq!(scenes.len(), 1, "All 20 should be one continuous burst");
        assert_eq!(scenes[0].image_ids.len(), 20);
    }
}
