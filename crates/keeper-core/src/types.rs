use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub type ImageId = u64;
pub type SceneId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: ImageId,
    pub file_path: PathBuf,
    pub file_name: String,
    /// Capture time in **milliseconds** since the Unix epoch (from EXIF).
    pub timestamp: Option<i64>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub focal_length_mm: Option<f32>,
    pub aperture: Option<f32>,
    pub iso: Option<u32>,
    /// Size of the embedded preview **as stored** (before rotation).
    pub preview_width: u32,
    pub preview_height: u32,
    /// EXIF orientation of the shot (1 = upright; 6 / 8 = rotated 90° CW /
    /// 270° CW). Embedded RAW previews are stored un-rotated, so this must be
    /// applied before analysis or display.
    pub orientation: u8,
    /// The embedded preview JPEG held in memory. Empty when the preview
    /// lives on disk instead (see `preview_file`).
    #[serde(skip)]
    pub preview_data: Vec<u8>,
    /// The preview JPEG on disk. The desktop app keeps previews only here —
    /// holding every full-size preview in RAM (~1.5 MB each) would need
    /// ~4.4 GB for a 3,000-image shoot.
    #[serde(skip)]
    pub preview_file: Option<PathBuf>,
}

impl ImageRecord {
    /// The preview JPEG bytes: borrowed from memory if held there, otherwise
    /// read from `preview_file`. Empty if the record has neither.
    pub fn load_preview(&self) -> std::io::Result<std::borrow::Cow<'_, [u8]>> {
        use std::borrow::Cow;
        if !self.preview_data.is_empty() {
            return Ok(Cow::Borrowed(&self.preview_data));
        }
        match &self.preview_file {
            Some(path) => std::fs::read(path).map(Cow::Owned),
            None => Ok(Cow::Borrowed(&[])),
        }
    }

    /// EXIF orientations 5–8 swap width and height.
    pub fn is_rotated_quarter_turn(&self) -> bool {
        (5..=8).contains(&self.orientation)
    }

    /// Preview size as the user should see it (after applying orientation).
    pub fn upright_preview_size(&self) -> (u32, u32) {
        if self.is_rotated_quarter_turn() {
            (self.preview_height, self.preview_width)
        } else {
            (self.preview_width, self.preview_height)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageScore {
    pub image_id: ImageId,
    pub sharpness_score: f64,
    pub has_face: bool,
    pub face_bbox: Option<BoundingBox>,
    pub blink_detected: bool,
    /// Blink score of the less-closed eye (0 = open, 1 = closed), if a face
    /// was analysed.
    pub blink_score: Option<f64>,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Keeper,
    Reject { reason: RejectReason },
    Unrated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    Blink,
    OutOfFocus,
    /// Nothing usable in the frame: black (lens cap, misfire) or blank
    /// (featureless wall, out-of-focus mush). Never chosen as a keeper.
    Unusable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub id: SceneId,
    pub image_ids: Vec<ImageId>,
    pub keeper_id: Option<ImageId>,
}
