use anyhow::Result;
use keeper_vision::preprocess;
use keeper_vision::runtime::OnnxModel;
use std::path::Path;

#[test]
fn test_mobilenet_loads_and_runs() -> Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/mobilenetv2-12.onnx");

    if !model_path.exists() {
        eprintln!(
            "SKIPPED: Model file not found at {:?}. \
             Download mobilenetv2-12.onnx to the models/ directory.",
            model_path
        );
        return Ok(());
    }

    let mut model = OnnxModel::load(&model_path, "MobileNetV2-test")?;
    println!("Model loaded successfully!");
    println!("   Inputs: {:?}", model.input_names());
    println!("   Outputs: {:?}", model.output_names());

    let test_image_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/test-photo.jpg");

    if !test_image_path.exists() {
        eprintln!(
            "SKIPPED: Test image not found at {:?}. \
             Place any .jpg file at test-data/test-photo.jpg.",
            test_image_path
        );
        return Ok(());
    }

    let jpeg_bytes = std::fs::read(&test_image_path)?;
    let img = preprocess::decode_jpeg(&jpeg_bytes)?;
    println!("Image decoded: {}x{}", img.width(), img.height());

    let resized = preprocess::resize_image(&img, 224, 224);
    let tensor = preprocess::image_to_nchw_tensor(
        &resized,
        preprocess::IMAGENET_MEAN,
        preprocess::IMAGENET_STD,
    );
    println!("Tensor created with shape: {:?}", tensor.shape());
    assert_eq!(tensor.shape(), &[1, 3, 224, 224]);

    let input_value = ort::value::Tensor::from_array(tensor)
        .map_err(|e| anyhow::anyhow!("Failed to create tensor: {e}"))?;
    let outputs = model
        .session
        .run(ort::inputs![input_value])
        .map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;

    println!("Inference completed! Got {} output(s).", outputs.len());

    let (output_shape, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output: {e}"))?;
    let shape: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();
    println!("   Output shape: {:?}", shape);

    let mut max_score: f32 = f32::NEG_INFINITY;
    let mut max_class: usize = 0;
    for (i, &score) in output_data.iter().enumerate() {
        if score > max_score {
            max_score = score;
            max_class = i;
        }
    }
    println!(
        "   Top prediction: class {} with score {:.4}",
        max_class, max_score
    );

    assert_eq!(shape[1], 1000);
    assert!(max_score.is_finite());

    println!("ONNX Runtime integration test PASSED!");

    Ok(())
}
