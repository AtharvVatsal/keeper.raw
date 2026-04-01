# Developer Setup Guide

This guide walks you through building keeper.raw from source on Windows, macOS, or Linux.

## Prerequisites

| Tool | Version | Installation |
|---|---|---|
| Rust | Stable (latest) | [rustup.rs](https://rustup.rs/) |
| Node.js | v18+ | [nodejs.org](https://nodejs.org/) |
| ExifTool | Latest | [exiftool.org](https://exiftool.org/) |
| Git | Latest | [git-scm.com](https://git-scm.com/) |

### Windows-specific

- **Visual Studio Build Tools** with the "Desktop development with C++" workload
- Install via: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### macOS-specific

- **Xcode Command Line Tools**: `xcode-select --install`

### Linux-specific

- System libraries for Tauri: https://v2.tauri.app/start/prerequisites/#linux

## Clone and Setup
```bash
git clone https://github.com/AtharvVatsal/keeper.raw
cd keeper.raw
npm install
```

## ONNX Models

The ML models are not included in the Git repo (they're too large). Download them:

1. Go to the [Releases page](https://github.com/AtharvVatsal/keeper.raw/releases) and find the "Model Files" asset
2. Download `yolov8n-face.onnx` and `face_landmark.onnx`
3. Place both files in the `models/` directory at the project root

Or follow the instructions in `models/README.md` to export them yourself.

## Development Commands
```bash
# Run in dev mode (hot-reload for frontend, recompile on Rust changes)
cargo tauri dev

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p keeper-stacker

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all

# Build a release binary
cargo tauri build
```

## Project Structure

keeper-raw/
├── src/                    # React/TypeScript frontend
│   └── App.tsx             # Main UI component
├── src-tauri/              # Tauri app shell + Rust commands
│   └── src/lib.rs          # IPC command handlers
├── crates/
│   ├── keeper-core/        # Shared types, config, errors
│   ├── keeper-ingest/      # File scanning, ExifTool, EXIF parsing
│   ├── keeper-stacker/     # Scene grouping (timestamp + pHash)
│   ├── keeper-vision/      # ML pipeline (face detect, focus score, blink)
│   └── keeper-xmp/         # XMP sidecar writer
├── models/                 # ONNX model files (not in Git)
├── docs/                   # Documentation
└── test-data/              # Test images

## Architecture

See [docs/architecture.md](architecture.md) for the full technical architecture.