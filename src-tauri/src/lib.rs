use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_vision::pipeline::{CullManifest, CullPipeline, CullProgress};
use keeper_xmp::{ExportEntry, ExportVerdict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{command, AppHandle, Emitter, Manager, State};

pub struct AppState {
    pub images: Mutex<Vec<ImageRecord>>,
    pub config: Mutex<CullConfig>,
}

#[derive(Debug, Serialize)]
pub struct ImageSummary {
    pub id: u64,
    pub file_name: String,
    pub timestamp: Option<i64>,
    pub camera_model: Option<String>,
    pub focal_length_mm: Option<f32>,
    pub iso: Option<u32>,
    pub preview_width: u32,
    pub preview_height: u32,
    pub preview_path: String,
}

#[derive(Debug, Deserialize)]
pub struct ExportImageVerdict {
    pub image_id: u64,
    pub verdict: String,
}

#[derive(Debug, Serialize)]
pub struct ExportResponse {
    pub files_written: usize,
    pub files_skipped: usize,
    pub errors: Vec<String>,
}

fn get_config_path() -> std::path::PathBuf {
    if let Some(config_dir) = dirs::config_dir() {
        let app_config_dir = config_dir.join("keeper-raw");
        let _ = fs::create_dir_all(&app_config_dir);
        return app_config_dir.join("config.toml");
    }
    std::path::PathBuf::from("keeper-raw-config.toml")
}

fn load_config_from_disk() -> CullConfig {
    let path = get_config_path();

    if path.exists() {
        match fs::read_to_string(&path) {
            Ok(contents) => match toml::from_str::<CullConfig>(&contents) {
                Ok(config) => {
                    eprintln!("Loaded config from: {}", path.display());
                    return config;
                }
                Err(e) => {
                    eprintln!(
                        "Warning: could not parse config file: {}. Using defaults.",
                        e
                    );
                }
            },
            Err(e) => {
                eprintln!(
                    "Warning: could not read config file: {}. Using defaults.",
                    e
                );
            }
        }
    }

    CullConfig::default()
}

fn save_config_to_disk(config: &CullConfig) -> Result<(), String> {
    let path = get_config_path();

    let toml_string =
        toml::to_string_pretty(config).map_err(|e| format!("Failed to serialize config: {}", e))?;

    fs::write(&path, &toml_string)
        .map_err(|e| format!("Failed to write config to '{}': {}", path.display(), e))?;

    eprintln!("Config saved to: {}", path.display());
    Ok(())
}

// =============================================================================
// Finding the AI models
//
// Installed app: the installer puts the models in the app's "resource"
// folder (listed in src-tauri/tauri.bundle.conf.json), next to the program on
// Windows and inside the .app bundle on macOS. Tauri tells us where that is.
//
// `cargo tauri dev`: nothing is copied, so debug builds also look in the
// repo's own models/ folder. Release builds never do — otherwise a broken
// installer would still work on the developer's PC and hide the problem.
// =============================================================================

/// Folders that may hold the models, in the order they're tried.
fn model_dir_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(resources) = app.path().resource_dir() {
        dirs.push(resources.join("models"));
    }
    if cfg!(debug_assertions) {
        dirs.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("models"),
        );
    }
    dirs
}

/// The first folder that has every model file the pipeline needs, or an
/// error saying where we looked and what was missing.
fn find_models_dir(candidates: &[PathBuf]) -> Result<PathBuf, String> {
    if let Some(dir) = candidates
        .iter()
        .find(|d| keeper_vision::pipeline::missing_model_files(d).is_empty())
    {
        return Ok(dir.clone());
    }
    let looked: Vec<String> = candidates
        .iter()
        .map(|d| {
            format!(
                "'{}' (missing: {})",
                d.display(),
                keeper_vision::pipeline::missing_model_files(d).join(", ")
            )
        })
        .collect();
    Err(format!(
        "The AI model files could not be found. Looked in: {}. \
         If you installed keeper.raw, please reinstall it. \
         Developers: see models/readme.md.",
        looked.join("; ")
    ))
}

// =============================================================================
// Finding ExifTool
//
// Same idea as the models: the installer ships ExifTool in the resource
// folder (fetched by scripts/fetch_exiftool.py). Debug builds also try the
// repo's vendor/exiftool/ and finally whatever `exiftool` is on the PATH.
// Release builds never use the PATH copy, so a missing bundled copy shows up
// on the developer's PC too instead of only on users' machines.
// =============================================================================

/// Why ExifTool can't be used, if it can't (checked once at start-up).
struct ExifToolProblem(Option<String>);

fn exiftool_dir_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(resources) = app.path().resource_dir() {
        dirs.push(resources.join("exiftool"));
    }
    if cfg!(debug_assertions) {
        dirs.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("vendor")
                .join("exiftool"),
        );
    }
    dirs
}

/// Picks the ExifTool to use for the whole session.
fn choose_exiftool(app: &AppHandle) -> ExifToolProblem {
    let candidates = exiftool_dir_candidates(app);
    match candidates
        .iter()
        .find_map(|d| keeper_ingest::ExifToolCommand::in_dir(d))
    {
        Some(command) => {
            keeper_ingest::use_exiftool(command);
            ExifToolProblem(None)
        }
        // `cargo tauri dev` without vendor/: fall back to the PATH.
        None if cfg!(debug_assertions) => ExifToolProblem(None),
        None => ExifToolProblem(Some(format!(
            "ExifTool, which keeper.raw uses to read RAW files, is missing from this \
             installation (looked in: {}). Please reinstall keeper.raw.",
            candidates
                .iter()
                .map(|d| format!("'{}'", d.display()))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

#[command]
async fn ingest_folder(
    path: String,
    state: State<'_, AppState>,
    exiftool: State<'_, ExifToolProblem>,
) -> Result<Vec<ImageSummary>, String> {
    if let Some(problem) = &exiftool.0 {
        return Err(problem.clone());
    }
    let dir = std::path::PathBuf::from(&path);

    if !dir.is_dir() {
        return Err(format!("Not a valid folder: {}", path));
    }

    // Previews live on disk only (one folder per opened folder), never in
    // RAM: ~1.5 MB each would mean ~4.4 GB for a 3,000-image shoot. The webview
    // displays these files and the pipeline reads them while analysing.
    let previews_root = std::env::temp_dir().join("keeper-raw-previews");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let preview_dir = previews_root.join(format!("ingest-{nanos}"));

    let records =
        keeper_ingest::ingest_directory_to(&dir, &preview_dir).map_err(|e| e.to_string())?;
    // Tag rotated shots so the webview displays them upright (embedded
    // previews have no EXIF of their own).
    keeper_ingest::tag_preview_files(&records).map_err(|e| e.to_string())?;

    // The new folder ingested fine: delete previews from earlier folders.
    if let Ok(entries) = fs::read_dir(&previews_root) {
        for entry in entries.flatten() {
            if entry.path() != preview_dir {
                let path = entry.path();
                let _ = if path.is_dir() {
                    fs::remove_dir_all(&path)
                } else {
                    fs::remove_file(&path)
                };
            }
        }
    }

    let summaries = records
        .iter()
        .map(|r| {
            let preview_path = r.preview_file.clone().unwrap_or_default();

            // Upright size: matches what the webview shows and the face boxes
            // the pipeline returns (both are in upright coordinates).
            let (preview_width, preview_height) = r.upright_preview_size();

            ImageSummary {
                id: r.id,
                file_name: r.file_name.clone(),
                timestamp: r.timestamp,
                camera_model: r.camera_model.clone(),
                focal_length_mm: r.focal_length_mm,
                iso: r.iso,
                preview_width,
                preview_height,
                preview_path: preview_path.to_string_lossy().to_string(),
            }
        })
        .collect();

    {
        let mut stored = state.images.lock().map_err(|e| e.to_string())?;
        *stored = records;
    }

    Ok(summaries)
}

#[command]
async fn cull_images(app: AppHandle, state: State<'_, AppState>) -> Result<CullManifest, String> {
    // Copy what the cull needs and release the lock straight away. Records
    // are small (previews live on disk), so this is cheap.
    let images: Vec<ImageRecord> = state.images.lock().map_err(|e| e.to_string())?.clone();

    if images.is_empty() {
        return Err("No images loaded. Open a folder first.".to_string());
    }

    let models_dir = find_models_dir(&model_dir_candidates(&app))?;

    let config = state.config.lock().map_err(|e| e.to_string())?.clone();

    // A cull of 3,000 images takes ~6 minutes. Run it on a background
    // thread so the window stays responsive (a non-async command would run
    // on the main thread and freeze the UI), and report progress as events.
    tauri::async_runtime::spawn_blocking(move || {
        let total = images.len();
        let _ = app.emit(CULL_PROGRESS_EVENT, CullProgress { analysed: 0, total });

        let pipeline = CullPipeline::new(&models_dir, config)
            .map_err(|e| format!("Failed to initialize pipeline: {}", e))?;

        // At most ~200 progress events per cull, plus the final one.
        let step = (total / 200).max(1);
        pipeline
            .run_with_progress(&images, &|progress| {
                if progress.analysed.is_multiple_of(step) || progress.analysed == progress.total {
                    let _ = app.emit(CULL_PROGRESS_EVENT, progress);
                }
            })
            .map_err(|e| format!("Pipeline failed: {}", e))
    })
    .await
    .map_err(|e| format!("Cull task failed: {}", e))?
}

/// Event emitted during a cull with a [`CullProgress`] payload
/// (`{ analysed, total }`); the frontend shows it as a progress bar.
const CULL_PROGRESS_EVENT: &str = "cull-progress";

#[command]
fn export_xmp(
    verdicts: Vec<ExportImageVerdict>,
    state: State<'_, AppState>,
) -> Result<ExportResponse, String> {
    let images = state.images.lock().map_err(|e| e.to_string())?;

    if images.is_empty() {
        return Err("No images loaded. Open a folder first.".to_string());
    }

    let image_map: HashMap<u64, &ImageRecord> = images.iter().map(|img| (img.id, img)).collect();

    let entries: Vec<ExportEntry> = verdicts
        .iter()
        .filter_map(|v| {
            let image = image_map.get(&v.image_id)?;
            let verdict = match v.verdict.as_str() {
                "keeper" => ExportVerdict::Keeper,
                "reject" => ExportVerdict::Reject,
                _ => ExportVerdict::Unrated,
            };
            Some(ExportEntry {
                raw_file_path: image.file_path.clone(),
                verdict,
            })
        })
        .collect();

    let result = keeper_xmp::export_xmp_sidecars(&entries);

    Ok(ExportResponse {
        files_written: result.files_written,
        files_skipped: result.files_skipped,
        errors: result
            .errors
            .iter()
            .map(|(path, err)| format!("{}: {}", path.display(), err))
            .collect(),
    })
}

#[command]
fn get_config(state: State<'_, AppState>) -> Result<CullConfig, String> {
    let config = state.config.lock().map_err(|e| e.to_string())?;
    Ok(config.clone())
}

#[command]
fn save_config(config: CullConfig, state: State<'_, AppState>) -> Result<(), String> {
    {
        let mut stored = state.config.lock().map_err(|e| e.to_string())?;
        *stored = config.clone();
    }

    save_config_to_disk(&config)?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let initial_config = load_config_from_disk();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            images: Mutex::new(vec![]),
            config: Mutex::new(initial_config),
        })
        .setup(|app| {
            let problem = choose_exiftool(app.handle());
            app.manage(problem);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ingest_folder,
            cull_images,
            export_xmp,
            get_config,
            save_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp folder containing the given model files (empty files are
    /// enough: only their presence is checked).
    fn folder_with(name: &str, files: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("keeper-raw-models-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for f in files {
            fs::write(dir.join(f), b"").unwrap();
        }
        dir
    }

    #[test]
    fn test_first_complete_folder_wins() {
        let all = keeper_vision::pipeline::REQUIRED_MODEL_FILES;
        let partial = folder_with("partial", &all[..2]);
        let complete = folder_with("complete", &all);
        let also_complete = folder_with("also-complete", &all);
        let found = find_models_dir(&[partial, complete.clone(), also_complete]).unwrap();
        assert_eq!(found, complete);
    }

    #[test]
    fn test_error_lists_every_folder_and_missing_file() {
        let all = keeper_vision::pipeline::REQUIRED_MODEL_FILES;
        let partial = folder_with("partial-only", &all[1..]);
        let missing = std::env::temp_dir().join("keeper-raw-models-test-does-not-exist");
        let err = find_models_dir(&[partial.clone(), missing.clone()]).unwrap_err();
        assert!(err.contains(&partial.display().to_string()));
        assert!(err.contains(&missing.display().to_string()));
        assert!(err.contains(all[0]), "{err}");
    }
}
