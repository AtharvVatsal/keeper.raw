//! keeper-xmp: XMP sidecar file writer for keeper.raw
//!
//! Writes industry-standard .xmp sidecar files next to RAW files.
//! These tiny XML files tell editing software (Lightroom, Darktable,
//! Capture One) what rating and label each image should have.
//!
//! Rating scheme:
//!   Keeper  → 5 stars (xmp:Rating = 5)
//!   Reject  → reject flag (xmp:Rating = -1, xmp:Label = "Reject")
//!   Unrated → no file written (leave the image untouched)

use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// The verdict for a single image, as decided by the AI + user overrides.
/// This is a simple enum that the Tauri command converts from the frontend data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportVerdict {
    Keeper,
    Reject,
    Unrated,
}

/// A single image to export, with its file path and verdict.
#[derive(Debug, Clone)]
pub struct ExportEntry {
    /// Path to the original RAW file (e.g., A:\photos\DSC_0571.NEF)
    pub raw_file_path: PathBuf,
    /// The final verdict (AI + user overrides combined)
    pub verdict: ExportVerdict,
}

/// Result of an XMP export operation.
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// How many .xmp files were written
    pub files_written: usize,
    /// How many images were skipped (unrated — no file needed)
    pub files_skipped: usize,
    /// Any files that failed to write (path + error message)
    pub errors: Vec<(PathBuf, String)>,
}

/// Write XMP sidecar files for all entries.
///
/// For each entry:
///   - Keeper → writes a .xmp file with Rating=5
///   - Reject → writes a .xmp file with Rating=-1, Label="Reject"
///   - Unrated → skips (no file written)
///
/// The .xmp file is placed next to the RAW file with the same name:
///   DSC_0571.NEF → DSC_0571.xmp
///
/// # Arguments
/// * `entries` — List of images with their verdicts
///
/// # Returns
/// An ExportResult with counts and any errors.
pub fn export_xmp_sidecars(entries: &[ExportEntry]) -> ExportResult {
    let mut files_written = 0;
    let mut files_skipped = 0;
    let mut errors: Vec<(PathBuf, String)> = Vec::new();

    info!("Exporting XMP sidecars for {} images...", entries.len());

    for entry in entries {
        match entry.verdict {
            ExportVerdict::Unrated => {
                files_skipped += 1;
                continue;
            }
            ExportVerdict::Keeper | ExportVerdict::Reject => {
                let xmp_path = get_xmp_path(&entry.raw_file_path);
                let xmp_content = generate_xmp_content(entry.verdict);

                match fs::write(&xmp_path, &xmp_content) {
                    Ok(()) => {
                        debug!("  Wrote: {}", xmp_path.display());
                        files_written += 1;
                    }
                    Err(e) => {
                        warn!("  Failed to write {}: {}", xmp_path.display(), e);
                        errors.push((xmp_path, e.to_string()));
                    }
                }
            }
        }
    }

    info!(
        "XMP export complete: {} written, {} skipped, {} errors",
        files_written,
        files_skipped,
        errors.len()
    );

    ExportResult {
        files_written,
        files_skipped,
        errors,
    }
}

/// Convert a RAW file path to its XMP sidecar path.
///
/// Example: "A:\photos\DSC_0571.NEF" → "A:\photos\DSC_0571.xmp"
fn get_xmp_path(raw_path: &Path) -> PathBuf {
    raw_path.with_extension("xmp")
}

/// Generate the XMP XML content for a given verdict.
///
/// This produces a minimal, valid XMP sidecar that is compatible with:
///   - Adobe Lightroom Classic 2024+
///   - Darktable 4.x
///   - Capture One 23+
///
/// The format follows the XMP specification (ISO 16684-1) using the
/// standard xmp: namespace for Rating and Label properties.
fn generate_xmp_content(verdict: ExportVerdict) -> String {
    let (rating, label_attr) = match verdict {
        ExportVerdict::Keeper => (5, String::new()),
        ExportVerdict::Reject => (
            -1,
            r#"
      xmp:Label="Reject""#
                .to_string(),
        ),
        ExportVerdict::Unrated => (0, String::new()),
    };

    format!(
        r#"<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmp:Rating="{rating}"
      xmp:CreatorTool="keeper.raw"{label_attr}>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Processed by keeper.raw</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_xmp_path_from_nef() {
        let raw = PathBuf::from("A:\\photos\\DSC_0571.NEF");
        let xmp = get_xmp_path(&raw);
        assert_eq!(xmp, PathBuf::from("A:\\photos\\DSC_0571.xmp"));
    }

    #[test]
    fn test_xmp_path_from_cr3() {
        let raw = PathBuf::from("/home/user/photos/IMG_1234.CR3");
        let xmp = get_xmp_path(&raw);
        assert_eq!(xmp, PathBuf::from("/home/user/photos/IMG_1234.xmp"));
    }

    #[test]
    fn test_keeper_xmp_contains_rating_5() {
        let content = generate_xmp_content(ExportVerdict::Keeper);
        assert!(content.contains(r#"xmp:Rating="5""#));
        assert!(!content.contains(r#"xmp:Label="Reject""#));
    }

    #[test]
    fn test_reject_xmp_contains_rating_negative_1() {
        let content = generate_xmp_content(ExportVerdict::Reject);
        assert!(content.contains(r#"xmp:Rating="-1""#));
        assert!(content.contains(r#"xmp:Label="Reject""#));
    }

    #[test]
    fn test_xmp_contains_xpacket() {
        let content = generate_xmp_content(ExportVerdict::Keeper);
        assert!(content.contains("<?xpacket begin="));
        assert!(content.contains("<?xpacket end="));
    }

    #[test]
    fn test_xmp_contains_creator_tool() {
        let content = generate_xmp_content(ExportVerdict::Keeper);
        assert!(content.contains(r#"xmp:CreatorTool="keeper.raw""#));
    }

    #[test]
    fn test_export_skips_unrated() {
        let temp = std::env::temp_dir().join("keeper-xmp-test");
        let _ = fs::create_dir_all(&temp);

        let raw_path = temp.join("test_skip.NEF");
        let _ = fs::write(&raw_path, b"fake raw");

        let entries = vec![ExportEntry {
            raw_file_path: raw_path.clone(),
            verdict: ExportVerdict::Unrated,
        }];

        let result = export_xmp_sidecars(&entries);
        assert_eq!(result.files_written, 0);
        assert_eq!(result.files_skipped, 1);

        let xmp_path = temp.join("test_skip.xmp");
        assert!(!xmp_path.exists());

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_export_writes_keeper() {
        let temp = std::env::temp_dir().join("keeper-xmp-test-write");
        let _ = fs::create_dir_all(&temp);

        let raw_path = temp.join("test_keeper.CR2");
        let _ = fs::write(&raw_path, b"fake raw");

        let entries = vec![ExportEntry {
            raw_file_path: raw_path.clone(),
            verdict: ExportVerdict::Keeper,
        }];

        let result = export_xmp_sidecars(&entries);
        assert_eq!(result.files_written, 1);
        assert_eq!(result.errors.len(), 0);

        let xmp_path = temp.join("test_keeper.xmp");
        assert!(xmp_path.exists());
        let content = fs::read_to_string(&xmp_path).unwrap();
        assert!(content.contains(r#"xmp:Rating="5""#));

        let _ = fs::remove_dir_all(&temp);
    }
}
