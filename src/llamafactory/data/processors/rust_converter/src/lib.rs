use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

mod models;
use models::*;

/// Convert single code sample to training format without Python context
fn convert_single(data: &str, format_type: &str, dataset_type: &str) -> Result<HashMap<String, Vec<Option<String>>>, String> {
    let mut result = HashMap::new();
    
    match dataset_type {
        "starcoder" => {
            let code_input: StarcoderInput = serde_json::from_str(data)
                .map_err(|e| format!("Invalid starcoder input data: {}", e))?;
            
            match format_type {
                "alpaca" => {
                    let converted = AlpacaFormat::from_starcoder(code_input);
                    result.insert("_prompt".to_string(), vec![Some(converted._tools.clone())]);
                    result.insert("_response".to_string(), vec![Some(converted._tools)]);
                    result.insert("_system".to_string(), vec![Some(converted._system)]);
                    result.insert("_tools".to_string(), vec![Some("".to_string())]);
                    result.insert("_images".to_string(), vec![None]);
                    result.insert("_videos".to_string(), vec![None]);
                }
                _ => return Err("Unsupported format type for starcoder".to_string()),
            }
        }
        "rust_explanation" => {
            let code_input: RustExplanationInput = serde_json::from_str(data)
                .map_err(|e| format!("Invalid rust explanation input data: {}", e))?;
            
            match format_type {
                "alpaca" => {
                    let converted = AlpacaFormat::from_rust_explanation(code_input);
                    result.insert("_prompt".to_string(), vec![Some(converted._tools.clone())]);
                    result.insert("_response".to_string(), vec![Some(converted._tools)]);
                    result.insert("_system".to_string(), vec![Some(converted._system)]);
                    result.insert("_tools".to_string(), vec![Some("".to_string())]);
                    result.insert("_images".to_string(), vec![None]);
                    result.insert("_videos".to_string(), vec![None]);
                }
                _ => return Err("Unsupported format type for rust explanation".to_string()),
            }
        }
        _ => return Err("Unsupported dataset type".to_string()),
    }

    Ok(result)
}

/// Convert code data to training format (Python interface)
#[pyfunction]
fn convert_code_data(_py: Python<'_>, data: String, format_type: String, dataset_type: String) -> PyResult<String> {
    let result = convert_single(&data, &format_type, &dataset_type)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    let output = DatasetOutput {
        data: result.into_iter().map(|(k, v)| (k, vec![v])).collect(),
        features: Features::default(),
    };

    serde_json::to_string(&output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e)))
}

/// Process multiple code samples efficiently using parallel processing
#[pyfunction]
fn batch_convert_code_data(_py: Python<'_>, data: Vec<String>, format_type: String, dataset_type: String) -> PyResult<String> {
    let format_type = Arc::new(format_type);
    let dataset_type = Arc::new(dataset_type);
    
    let results: Result<Vec<_>, _> = data.par_iter()
        .map(|sample| {
            let format_ref = format_type.as_ref();
            let dataset_ref = dataset_type.as_ref();
            convert_single(sample, format_ref, dataset_ref)
        })
        .collect();
    
    let results = results.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    let mut combined = HashMap::new();
    for key in ["_prompt", "_response", "_system", "_tools", "_images", "_videos"] {
        let values: Vec<_> = results.iter()
            .filter_map(|r| r.get(key))
            .cloned()
            .collect();
        combined.insert(key.to_string(), values);
    }
    
    let output = DatasetOutput {
        data: combined,
        features: Features::default(),
    };

    serde_json::to_string(&output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e)))
}

/// Python module initialization
#[pymodule]
fn rust_converter(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Data conversion functions
    m.add_function(wrap_pyfunction!(convert_code_data, py)?)?;
    m.add_function(wrap_pyfunction!(batch_convert_code_data, py)?)?;

    // Tokenization functions
    m.add_function(wrap_pyfunction!(fast_tokenize, py)?)?;
    m.add_function(wrap_pyfunction!(fast_decode, py)?)?;

    // Dataset packing functions
    m.add_function(wrap_pyfunction!(pack_dataset, py)?)?;

    Ok(())
}
