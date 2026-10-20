use keeper_core::config::CullConfig;
use keeper_core::types::ImageRecord;
use keeper_vision::pipeline::{CullManifest, CullPipeline};
use keeper_xmp::{ExportEntry, ExportVerdict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;
use tauri::{command, State};

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

#[command]
async fn ingest_folder(
    path: String,
    state: State<'_, AppState>,
) -> Result<Vec<ImageSummary>, String> {
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
fn cull_images(state: State<'_, AppState>) -> Result<CullManifest, String> {
    let images = state.images.lock().map_err(|e| e.to_string())?;

    if images.is_empty() {
        return Err("No images loaded. Open a folder first.".to_string());
    }

    // Dev builds run from src-tauri/, hence "../models" (known limitation —
    // installed builds will need the Tauri resource path instead).
    let models_dir = std::path::PathBuf::from("../models");
    let missing = keeper_vision::pipeline::missing_model_files(&models_dir);
    if !missing.is_empty() {
        let cwd = std::env::current_dir().unwrap_or_default();
        return Err(format!(
            "Missing model file(s) in '{}': {}. Current directory: '{}'. \
             See models/readme.md, and run 'cargo tauri dev' from the project root folder.",
            models_dir.display(),
            missing.join(", "),
            cwd.display()
        ));
    }

    let config = state.config.lock().map_err(|e| e.to_string())?.clone();

    let pipeline = CullPipeline::new(&models_dir, config)
        .map_err(|e| format!("Failed to initialize pipeline: {}", e))?;

    let manifest = pipeline
        .run(&images)
        .map_err(|e| format!("Pipeline failed: {}", e))?;

    Ok(manifest)
}

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
