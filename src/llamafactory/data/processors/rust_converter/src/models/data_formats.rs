use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeInput {
    pub content: String,
    pub role: String,
}

#[derive(Debug, Serialize)]
pub struct AlpacaFormat {
    pub _system: String,
    pub _tools: String,
}

#[derive(Debug, Serialize)]
pub struct ShareGPTFormat {
    pub _system: String,
    pub _tools: String,
}

#[derive(Debug, Deserialize)]
pub struct StarcoderInput {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct RustExplanationInput {
    pub input: String,
    pub output: String,
}

impl AlpacaFormat {
    pub fn from_code(content: String) -> Self {
        Self {
            _system: "You are a helpful coding assistant.".to_string(),
            _tools: content,
        }
    }

    pub fn from_starcoder(input: StarcoderInput) -> Self {
        Self {
            _system: "You are a helpful coding assistant.".to_string(),
            _tools: input.content,
        }
    }

    pub fn from_rust_explanation(input: RustExplanationInput) -> Self {
        Self {
            _system: "You are a helpful coding assistant.".to_string(),
            _tools: format!("Input:\n{}\nOutput:\n{}", input.input, input.output),
        }
    }
}

impl ShareGPTFormat {
    pub fn from_code(content: String) -> Self {
        Self {
            _system: "You are a helpful coding assistant.".to_string(),
            _tools: content,
        }
    }
}
