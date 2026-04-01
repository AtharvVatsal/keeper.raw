# Architecture Guide

This document describes the internal architecture of keeper-raw for developers.

## Overview

keeper-raw follows a modular architecture with clear separation between:

1. **Frontend** (React/TypeScript) - User interface
2. **Tauri Bridge** (Rust) - IPC between frontend and backend
3. **Backend** (Rust crates) - Core business logic

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│                    (src/App.tsx)                        │
└───────────────────────┬─────────────────────────────────┘
                        │ Tauri IPC (invoke)
┌───────────────────────▼─────────────────────────────────┐
│                  Tauri Commands                         │
│              (src-tauri/src/lib.rs)                    │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  Ingest   │   │  Stacker  │   │   Vision  │
│           │   │           │   │           │
└─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │               │
      └───────────────┼───────────────┘
                      ▼
              ┌───────────────┐
              │     XMP      │
              │   Exporter    │
              └───────────────┘
```

## Data Flow

### 1. Ingest Phase

```
RAW Files → ExifTool → JPEG Preview + EXIF → ImageRecord[]
```

**Key files:**
- `crates/keeper-ingest/src/scanner.rs` - Finds RAW files in directory
- `crates/keeper-ingest/src/extractor.rs` - Extracts previews using ExifTool

**Supported formats:** CR2, CR3, NEF, ARW, RAF, DNG, ORF, RW2

### 2. Scene Stacking

```
ImageRecord[] → Timestamp Clustering → pHash Splitting → Scene[]
```

**Algorithm:**
1. Group images by timestamp (default: within 3 seconds)
2. Split groups using perceptual hashing (pHash) when visual similarity exceeds threshold

**Key files:**
- `crates/keeper-stacker/src/stacker.rs` - Main stacking logic
- `crates/keeper-stacker/src/hasher.rs` - pHash computation

### 3. Vision Pipeline

For each image in a scene:

```
Image → Face Detection → Focus Scoring → Blink Detection → ImageScore
```

**Stage 1: Face Detection**
- Model: YOLOv8-face (ONNX)
- Input: 640x640 RGB
- Output: Bounding boxes + confidence scores
- Post-processing: NMS (Non-Maximum Suppression)

**Stage 2: Focus Scoring**
- Algorithm: Laplacian variance
- Region: Eye region (if face detected) or center (fallback)
- Formula: `sharpness = variance(laplacian(gray)) / noise_estimate`

**Stage 3: Blink Detection**
- Model: FaceMesh (ONNX) - 468 facial landmarks
- Metric: Eye Aspect Ratio (EAR)
- Threshold: EAR < 0.21 indicates blink

**Key files:**
- `crates/keeper-vision/src/pipeline.rs` - Pipeline orchestration
- `crates/keeper-vision/src/face_detector.rs` - YOLOv8-face wrapper
- `crates/keeper-vision/src/focus_scorer.rs` - Sharpness analysis
- `crates/keeper-vision/src/blink_detector.rs` - EAR computation
- `crates/keeper-vision/src/preprocess.rs` - Image preprocessing

### 4. Keeper Selection

For each scene:
1. Filter out images with detected blinks
2. Select image with highest sharpness score
3. If all images have blinks, select sharpest anyway

### 5. Export

```
Verdicts → XMP Sidecar Files
```

- Keeper: Rating = 5, no label
- Reject: Rating = -1, Label = "Reject"
- Unrated: No file written

**Key files:**
- `crates/keeper-xmp/src/lib.rs` - XMP generation

## Module Reference

### keeper-core

**Purpose:** Shared types and configuration

**Files:**
- `types.rs` - `ImageRecord`, `ImageScore`, `Scene`, `Verdict`
- `config.rs` - `CullConfig` with thresholds
- `error.rs` - Error types

### keeper-ingest

**Purpose:** RAW file ingestion and preview extraction

**Key functions:**
- `scan_directory(path) -> Vec<PathBuf>` - Find RAW files
- `ingest_directory(path) -> Vec<ImageRecord>` - Full ingestion

### keeper-stacker

**Purpose:** Scene grouping using time + visual similarity

**Key functions:**
- `stack_scenes(images, config) -> Vec<Scene>` - Main entry point

### keeper-vision

**Purpose:** AI-powered image analysis

**Key functions:**
- `CullPipeline::new(face_model, mesh_model, config)` - Initialize
- `CullPipeline::run(images) -> CullManifest` - Run full pipeline

### keeper-xmp

**Purpose:** XMP sidecar file generation

**Key functions:**
- `export_xmp_sidecars(entries) -> ExportResult` - Write XMP files

## Tauri Commands

| Command | Parameters | Returns |
|---------|------------|---------|
| `ingest_folder` | `path: String` | `Vec<ImageSummary>` |
| `cull_images` | - | `CullManifest` |
| `export_xmp` | `verdicts: Vec<ExportImageVerdict>` | `ExportResponse` |
| `get_config` | - | `CullConfig` |
| `save_config` | `config: CullConfig` | `()` |

## State Management

**Frontend (React):**
- `images: ImageSummary[]` - Loaded images
- `manifest: CullManifest` - Cull results
- `overrides: Map<id, verdict>` - Manual overrides
- `config: CullConfig` - Current settings

**Backend (Rust):**
- `AppState::images` - Full ImageRecord with preview data
- `AppState::config` - Persisted configuration

## Configuration

Settings are persisted to TOML files:

```toml
burst_threshold_secs = 3.0
phash_similarity_threshold = 14
blink_ear_threshold = 0.21
face_detection_confidence = 0.5
```

## Performance Considerations

1. **Preview Extraction**: Sequential using ExifTool (slowest phase)
2. **Scene Stacking**: Parallel pHash computation using Rayon
3. **Vision Pipeline**: Sequential per image (models loaded once)
4. **Memory**: Previews stored in memory (~1-5MB each)

## Testing

```bash
# Unit tests per crate
cargo test -p keeper-core
cargo test -p keeper-stacker
cargo test -p keeper-vision

# Integration tests
cargo test --all
```

## Adding New Features

### New Vision Algorithm

1. Add algorithm in `crates/keeper-vision/src/`
2. Update `pipeline.rs` to call new algorithm
3. Add configuration options if needed in `keeper-core`

### New Export Format

1. Add crate in `crates/`
2. Add Tauri command in `src-tauri/src/lib.rs`
3. Call from frontend in `src/App.tsx`
