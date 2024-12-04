use pyo3::prelude::*;
use std::sync::Arc;
use tokenizers::tokenizer::{Tokenizer, Result as TokenizerResult};

pub struct FastTokenizer {
    tokenizer: Arc<Tokenizer>,
}

impl FastTokenizer {
    pub fn new(tokenizer_path: &str) -> TokenizerResult<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
        })
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> PyResult<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[pyfunction]
pub fn fast_tokenize(tokenizer_path: &str, text: &str) -> PyResult<Vec<u32>> {
    let tokenizer = FastTokenizer::new(tokenizer_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    tokenizer.encode(text)
}

#[pyfunction]
pub fn fast_decode(tokenizer_path: &str, ids: Vec<u32>) -> PyResult<String> {
    let tokenizer = FastTokenizer::new(tokenizer_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    tokenizer.decode(&ids)
}
