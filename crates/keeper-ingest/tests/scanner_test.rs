use std::path::{Path, PathBuf};

/// Folder of real RAW files for the slow tests. Set it in PowerShell with
/// `$env:KEEPER_TEST_RAWS = "A:\Work\keeper-raw\TestRAWs"`; unset = skip.
fn test_raw_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("KEEPER_TEST_RAWS").ok()?);
    if dir.is_dir() {
        Some(dir)
    } else {
        println!(
            "Skipping: KEEPER_TEST_RAWS={} is not a folder.",
            dir.display()
        );
        None
    }
}

fn exiftool_available() -> bool {
    std::process::Command::new("exiftool")
        .arg("-ver")
        .output()
        .is_ok()
}

#[test]
fn test_scan_finds_raw_files() {
    let Some(test_dir) = test_raw_dir() else {
        println!("Skipping: set KEEPER_TEST_RAWS to run.");
        return;
    };

    let files = keeper_ingest::scan_directory(&test_dir).unwrap();
    println!("Found {} RAW files", files.len());
    assert!(!files.is_empty());
}

#[test]
fn test_ingest_extracts_previews() {
    let Some(test_dir) = test_raw_dir() else {
        println!("Skipping: set KEEPER_TEST_RAWS to run.");
        return;
    };
    let raw_count = keeper_ingest::scan_directory(&test_dir).unwrap().len();

    let records = keeper_ingest::ingest_directory(&test_dir).unwrap();
    assert_eq!(records.len(), raw_count, "every RAW should yield a preview");
    println!("Extracted {} previews", records.len());

    for record in &records {
        println!(
            "  {} — {}x{} — ISO {:?} — {:?}",
            record.file_name,
            record.preview_width,
            record.preview_height,
            record.iso,
            record.camera_model,
        );
        assert!(
            !record.preview_data.is_empty(),
            "Preview should not be empty for {}",
            record.file_name
        );
    }
}

#[test]
fn test_missing_file_is_skipped_not_fatal() {
    if !exiftool_available() {
        println!("Skipping: exiftool not installed.");
        return;
    }
    let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("does-not-exist.NEF");
    let records = keeper_ingest::extractor::extract_all(&[missing]).unwrap();
    assert!(records.is_empty());
}

#[test]
fn test_empty_input_needs_no_exiftool() {
    assert!(keeper_ingest::extractor::extract_all(&[])
        .unwrap()
        .is_empty());
}
