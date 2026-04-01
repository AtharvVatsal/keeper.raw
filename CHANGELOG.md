# Changelog

All notable changes to keeper-raw will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-04-01

### Added

- Initial release
- RAW file support (CR2, CR3, NEF, ARW, RAF, DNG, ORF, RW2)
- Face detection using YOLOv8-face ONNX model
- Focus scoring using Laplacian variance
- Blink detection using FaceMesh landmarks
- Scene grouping via timestamp clustering and pHash
- Automatic keeper selection
- Manual override (keeper/reject/unrate)
- XMP sidecar export for Lightroom, Darktable, Capture One
- Grid view with scene cards
- Loupe view with face-centered zoom
- Keyboard shortcuts (K, U, X, arrows, Z, Esc)
- Filter by All/Keepers/Rejects
- Settings panel with configurable thresholds
- Persistent configuration storage

### Technology

- Tauri 2 desktop framework
- React 19 + TypeScript frontend
- Rust backend with modular crates
- ONNX Runtime for ML inference

---

## Known Limitations

1. **Preview Extraction Speed**: Using ExifTool sequentially; future versions may parallelize
2. **RAW Format Support**: Depends on ExifTool's capabilities
3. **Memory Usage**: Large batches may require significant RAM
4. **GPU Acceleration**: Currently CPU-only; GPU support planned

---

## Migration Guide

### From v0.1.0

First release - no migration needed.

---

## Credits

Thank you to all contributors!

- YOLOv8-face: [akanuasia](https://github.com/akanuasia/YOLOv8-face)
- FaceMesh: Google MediaPipe
- Built with Tauri, React, ONNX Runtime
