use thiserror::Error;

#[derive(Error, Debug)]
pub enum KeeperError {
    #[error("Failed to read file: {path}")]
    FileReadError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("No preview found in RAW file: {path}")]
    NoPreviewError { path: String },

    #[error("Could not parse EXIF data: {path}")]
    ExifError { path: String },

    #[error("AI model failed: {details}")]
    InferenceError { details: String },

    #[error("Could not write XMP file: {path}")]
    XmpWriteError {
        path: String,
        #[source]
        source: std::io::Error,
    },
}