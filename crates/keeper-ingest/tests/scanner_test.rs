use std::path::Path;

#[test]
fn test_scan_finds_raw_files() {
    let test_dir = Path::new(r"A:\TestRAWs");
    if !test_dir.exists() {
        println!("Skipping: test folder not found.");
        return;
    }

    let files = keeper_ingest::scan_directory(test_dir).unwrap();
    println!("Found {} RAW files", files.len());
    assert!(!files.is_empty());
}

#[test]
fn test_ingest_extracts_previews() {
    let test_dir = Path::new(r"A:\TestRAWs");
    if !test_dir.exists() {
        println!("Skipping: test folder not found.");
        return;
    }

    let records = keeper_ingest::ingest_directory(test_dir).unwrap();
    println!("Extracted {} previews", records.len());

    for record in &records {
        println!("  {} — {}x{} — ISO {:?} — {:?}",
            record.file_name,
            record.preview_width,
            record.preview_height,
            record.iso,
            record.camera_model,
        );
        assert!(!record.preview_data.is_empty(),
            "Preview should not be empty for {}", record.file_name);
    }
}