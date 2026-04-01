use anyhow::{anyhow, Result};
use ort::session::Session;
use std::path::Path;
use tracing::info;

pub struct OnnxModel {
    pub session: Session,
    pub name: String,
}

impl OnnxModel {
    pub fn load(model_path: &Path, name: &str) -> Result<Self> {
        info!("Loading ONNX model '{}' from {:?}", name, model_path);

        let session = Session::builder()
            .map_err(|e| anyhow!("Failed to create ONNX session builder: {e}"))?
            .with_intra_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            )
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
