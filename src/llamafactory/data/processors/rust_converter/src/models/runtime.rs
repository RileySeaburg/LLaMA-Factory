use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use std::sync::Arc;
use std::collections::HashMap;
use super::features::{Features, Value, FeatureType, MessageSchema};

pub struct FeatureAdjuster {
    runtime: Runtime,
    tx: mpsc::Sender<FeatureAdjustment>,
}

enum FeatureAdjustment {
    UpdateResponse(ResponseType),
}

enum ResponseType {
    Sequence,
    MessageSchema,
}

impl FeatureAdjuster {
    pub fn new() -> Self {
        let runtime = Runtime::new().unwrap();
        let (tx, mut rx) = mpsc::channel(32);
        
        let tx_clone = tx.clone();
        runtime.spawn(async move {
            while let Some(adjustment) = rx.recv().await {
                match adjustment {
                    FeatureAdjustment::UpdateResponse(resp_type) => {
                        match resp_type {
                            ResponseType::Sequence => {
                                // Signal that features were updated
                                let _ = tx_clone.send(FeatureAdjustment::UpdateResponse(ResponseType::Sequence)).await;
                            },
                            ResponseType::MessageSchema => {
                                // Signal that features were updated
                                let _ = tx_clone.send(FeatureAdjustment::UpdateResponse(ResponseType::MessageSchema)).await;
                            }
                        }
                    }
                }
            }
        });

        Self { runtime, tx }
    }

    pub async fn adjust_features(&self, error: &str) -> Features {
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

        let mut features = Features::default();
        if error.contains("Sequence(feature=Value(dtype='null'") {
            let _ = self.tx.send(FeatureAdjustment::UpdateResponse(ResponseType::Sequence)).await;
            features._response = vec![message_schema];
        }
        features
    }
}
