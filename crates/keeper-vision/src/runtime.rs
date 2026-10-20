use anyhow::{anyhow, Result};
use ort::session::Session;
use std::path::Path;
use tracing::info;

pub struct OnnxModel {
    pub session: Session,
    pub name: String,
}

/// Number of logical CPU cores (4 if unknown).
pub fn available_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

impl OnnxModel {
    /// Loads a model whose session may use every core.
    pub fn load(model_path: &Path, name: &str) -> Result<Self> {
        Self::load_with_threads(model_path, name, available_cores())
    }

    /// Loads a model whose session uses at most `intra_threads` threads.
    /// When several sessions run side by side (one per pipeline worker),
    /// giving each all cores would oversubscribe the CPU.
    pub fn load_with_threads(model_path: &Path, name: &str, intra_threads: usize) -> Result<Self> {
        info!(
            "Loading ONNX model '{}' from {:?} ({} thread(s))",
            name, model_path, intra_threads
        );

        let session = Session::builder()
            .map_err(|e| anyhow!("Failed to create ONNX session builder: {e}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow!("Failed to set thread count: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to load ONNX model from {:?}: {e}", model_path))?;

        info!("Model '{}' loaded successfully.", name);

        let num_inputs = session.inputs().len();
        let num_outputs = session.outputs().len();
        info!("  {} input(s), {} output(s)", num_inputs, num_outputs);

        for input in session.inputs() {
            info!("  Input: '{}'", input.name());
        }
        for output in session.outputs() {
            info!("  Output: '{}'", output.name());
        }

        Ok(OnnxModel {
            session,
            name: name.to_string(),
        })
    }

    pub fn input_names(&self) -> Vec<String> {
        let inputs = self.session.inputs();
        (0..inputs.len())
            .map(|i| inputs[i].name().to_string())
            .collect()
    }

    pub fn output_names(&self) -> Vec<String> {
        let outputs = self.session.outputs();
        (0..outputs.len())
            .map(|i| outputs[i].name().to_string())
            .collect()
    }
}
