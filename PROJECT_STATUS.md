# keeper.raw — Project Status

## Project Overview, Architecture & Status

**Atharv Vatsal · October 3, 2026**

This document pulls together the master file, the roadmap, and all five working sessions on keeper.raw (Phase 0 in March through the Phase 3 session that ran into June).

---

## 1. What it is

keeper.raw is a free, MIT-licensed, fully offline desktop app that does the first pass of photo culling for you. You point it at a folder of RAW files and it groups bursts into Scenes, picks the sharpest frame in each one (judged by eye sharpness when there's a face), flags blinks as rejects, and writes standard `.xmp` sidecars. When you open the folder in Lightroom, Darktable or Capture One, the ratings are already there.

It's aimed at wedding, event and portrait shooters who come home with thousands of frames. The pitch against Aftershoot and Narrative Select is that it has no subscription, nothing gets uploaded, and it runs on an ordinary laptop's CPU. It is built solo on Windows 11, and the repo is at `github.com/AtharvVatsal/keeper.raw`. The brand is the lowercase rounded wordmark **"keeper"** with **".raw"** tucked under the right side, in black and white variants.

---

## 2. System architecture

The app is a Tauri v2 shell with a React/TypeScript frontend. Behind it is a Rust workspace of five library crates and one benchmark binary, `keeper-bench`. The frontend talks to Rust only through Tauri commands, and culled results are held in an `AppState` behind a `Mutex`, so overrides and export work on a single source of truth.

### 2.1 The pipeline, stage by stage

1. **Ingest (`keeper-ingest`)**
   - Scans the folder for RAW files: officially CR2/CR3/NEF/ARW, with RAF/DNG/ORF/RW2 accepted as best effort.
   - Uses ExifTool to pull each file's embedded JPEG preview.
   - Parses EXIF: timestamp, camera make and model, focal length, aperture, ISO.
   - Using the embedded preview instead of decoding the RAW is the biggest performance decision in the app.

2. **Stacking (`keeper-stacker`)**
   - First clusters images by timestamp gap.
   - Then splits clusters further by perceptual-hash distance.
   - This second step catches a photographer who shoots continuously across two different compositions.
   - Real-world testing showed the roadmap defaults were too strict, so they were retuned:
     - Burst threshold: **1.5s → 3.0s**
     - pHash similarity: **10 → 14**

3. **Vision (`keeper-vision`)**
   - Everything runs through ONNX Runtime via `ort 2.0.0-rc.12`.
   - **Face detection:** YOLOv8-nano, with letterbox preprocessing and non-maximum suppression (NMS).
   - **Focus scoring:** Variance of Laplacian on the eye region, divided by a noise floor (MAD on a flat patch) so high-ISO noise doesn't look like sharpness. Frames with no face are scored on the centre of the frame.
   - **Blink detection:** MediaPipe FaceMesh V2 + Blendshape V2 eye-blink scores (replaced an Eye Aspect Ratio heuristic in October 2026). A shot is rejected only when the model is confident it sees a face and both eyes score as closed (≥ 0.40).
   - **Orchestration:** `CullPipeline` runs all of this and emits a JSON manifest.

4. **Export (`keeper-xmp`)**
   - Keepers get `Rating=5`.
   - Rejects get `Rating=-1` plus `Label=Reject`.
   - Everything else is left untouched.

### 2.2 The user interface

- A scene grid with collapsible burst stacks and count badges
- A full-screen loupe that auto-zooms to the detected face
- Arrow-key navigation between and within scenes
- `K` / `U` / `X` shortcuts to promote a keeper, unrate, or toggle reject
- A settings modal with sliders for the four thresholds, persisted to a TOML file

---

## 3. How much is done

| Phase | Status |
|---|---|
| **Phase 0 — Foundation** (scaffold, ingest, EXIF, thumbnail grid) | Done; passed a 15-test validation suite |
| **Phase 1 — The Brain** (ONNX, stacking, face, focus, blink, CullPipeline) | Done; 22+ tests, CLI produces a valid `cull_manifest.json` |
| **Phase 2 — The Experience** (UI wiring, loupe, overrides, XMP, settings) | Done, code-complete |
| **Phase 3 — Launch & Hardening** | Steps 1–3 done (README and docs, community files, `keeper-bench`); steps 4–7 not started |

By roadmap step count, that's roughly **85% of v0.1.0**. A more honest estimate is closer to **70–75%**. The remaining work is the least familiar part (CI and cross-platform packaging), and none of the MVP success criteria have actually been measured yet. The last recorded session on the project was in June 2026, so any CI work done since then is not reflected here.

---

## 4. What to flag before launch

### 4.1 Model licensing (most urgent)

Ultralytics YOLOv8 is AGPL-3.0, and most YOLOv8-face weights derive from it. Bundling those weights inside an MIT-licensed installer is a real conflict. Fix this before going public, not after.

Two cleaner face detectors to look at:

- **BlazeFace:** Apache 2.0, and it's from the same MediaPipe family as the Face Mesh model.
- **YuNet:** from the OpenCV model zoo.

Verify each license directly before switching.

### 4.2 ExifTool packaging

The stated goal is a single installer with no setup. ExifTool is an external program, so it has to ship inside the app (Tauri's `sidecar` / `externalBin` feature handles this), and the Perl dependency needs handling on macOS and Linux.

There's also a speed issue: launching ExifTool once per file is slow. The `-stay_open` batch mode, or eventually a pure-Rust preview extractor, would likely be needed to get near the 90-second ingest target.

### 4.3 Pipeline speed

`FaceDetector` and `BlinkDetector` sit behind a `Mutex` because `ort` needs `&mut self`. That means the ML stages run one image at a time; Rayon is not used in the pipeline yet.

A small pool of sessions, one per worker thread, would remove that bottleneck. The only real numbers so far are from the 26-image dev-profile run, about **108 seconds total**. The MVP target works out to roughly **160 ms per image**, end to end.

Run `keeper-bench` in release mode on **1,000+ real RAWs**; that's the first thing Step 6 needs to answer.

### 4.4 Accuracy is unvalidated

The roadmap sets two targets that need labeled data:

- The keeper matches a photographer's top pick at least **70%** of the time.
- Blink detection has at least **95% precision**.

The Phase 2 exit gate also called for five beta testers, which hasn't happened yet. Even three photographer friends running their own shoots would surface most problems.

### 4.5 Real-app XMP behavior

Lightroom Classic likely keeps pick/reject flags in its catalog rather than reading them from XMP. If so, the `-1` rating may not show up as a proper "Rejected" flag there.

Sidecar naming conventions also differ between apps (`IMG_0001.xmp` vs `IMG_0001.CR3.xmp`). Test both by actually importing into all three apps.

### 4.6 Code signing

Unsigned builds will trigger Windows SmartScreen and macOS Gatekeeper warnings. macOS notarization needs a paid Apple developer account.

Launching unsigned with clear "how to open this" instructions is fine for v0.1.0.

### 4.7 Smaller items

- The frontend is a single `App.tsx`. Splitting it into components before inviting contributors will make "good first issue" UI tasks much easier.
- Most testing used `.JPG` files. The roadmap's own top risk is that embedded preview sizes vary across camera bodies, so checking real CR3, NEF and ARW files is worth doing before launch.

---

## 5. What's next

Step 4 is a GitHub Actions CI workflow running `cargo fmt`, `clippy`, and tests on every push. The catch is that the ONNX-dependent tests need models the repo deliberately doesn't contain, so those should be skipped in CI.

After that come:

- **Step 5:** A tag-triggered release workflow that builds the `.msi`, `.dmg` and `.AppImage`
- **Step 6:** QA against the MVP success criteria
- **Step 7:** Going public and posting to `r/photography`, `r/WeddingPhotography` and Hacker News

The model-license fix should slot in before Step 5, since it decides what goes into the installer.

