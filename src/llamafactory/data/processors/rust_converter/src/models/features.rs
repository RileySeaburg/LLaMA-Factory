use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum FeatureType {
    String,
    Null,
}

#[derive(Debug, Serialize, Clone)]
pub struct Value {
    pub dtype: FeatureType,
    pub id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct MessageSchema {
    pub content: Value,
    pub role: Value,
}

#[derive(Debug, Serialize)]
pub struct Sequence {
    pub feature: Value,
    pub length: i32,
    pub id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Features {
    pub _prompt: Vec<MessageSchema>,
    pub _response: Vec<MessageSchema>,
    pub _system: Value,
    pub _tools: Value,
    pub _images: Value,
    pub _videos: Value,
}

impl Default for Features {
    fn default() -> Self {
        let string_value = Value {
            dtype: FeatureType::String,
            id: None,
        };
        let null_value = Value {
            dtype: FeatureType::Null,
            id: None,
        };
        let message_schema = MessageSchema {
            content: string_value.clone(),
            role: string_value.clone(),
        };

        Self {
            _prompt: vec![message_schema.clone()],
            _response: vec![message_schema],
            _system: string_value.clone(),
            _tools: string_value,
            _images: null_value.clone(),
            _videos: null_value,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DatasetOutput {
    pub data: HashMap<String, Vec<Vec<Option<String>>>>,
    pub features: Features,
}

impl DatasetOutput {
    pub fn ensure_message_schema(&mut self) {
        // Convert any sequence format to message schema format
        let string_value = Value {
            dtype: FeatureType::String,
            id: None,
        };
        let message_schema = MessageSchema {
            content: string_value.clone(),
            role: string_value,
        };

        if self.features._response.is_empty() {
            self.features._response = vec![message_schema];
        }
    }
}
