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
2. Download `yolov8n-face.onnx`, `face_landmarks_detector.onnx` and `face_blendshapes.onnx`
3. Place both files in the `models/` directory at the project root

Or follow the instructions in `models/readme.md` to export them yourself.

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

# Build the installers (models from models/ are packed inside)
npm run package
```

`npm run package` runs `tauri build` with `src-tauri/tauri.bundle.conf.json`, which lists the
model files to put in the installer. They're kept out of the main `tauri.conf.json` because the
models aren't in git: listing them there would make every build without them (e.g. CI) fail.
Installers land in `target/release/bundle/` (`.msi` and `-setup.exe` on Windows).

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

See [architecture.md](../architecture.md) for the full technical architecture.