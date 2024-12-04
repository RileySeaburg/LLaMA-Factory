use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct PackedBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub labels: Vec<Vec<i32>>,
}

impl PackedBatch {
    pub fn new() -> Self {
        Self {
            input_ids: Vec::new(),
            attention_mask: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn to_tuple(self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<i32>>) {
        (self.input_ids, self.attention_mask, self.labels)
    }
}

pub fn pack_sequences(
    sequences: Vec<Vec<u32>>,
    max_length: usize,
    pad_token_id: u32,
) -> PyResult<PackedBatch> {
    // Group sequences by similar lengths for more efficient packing
    let mut length_groups: HashMap<usize, Vec<Vec<u32>>> = HashMap::new();
    for seq in sequences {
        let len = seq.len();
        length_groups.entry(len).or_default().push(seq);
    }

    // Process each length group in parallel
    let packed_groups: Vec<_> = length_groups
        .par_iter()
        .map(|(_, group)| pack_length_group(group, max_length, pad_token_id))
        .collect::<PyResult<Vec<_>>>()?;

    // Combine results
    let mut result = PackedBatch::new();
    for mut group in packed_groups {
        result.input_ids.append(&mut group.input_ids);
        result.attention_mask.append(&mut group.attention_mask);
        result.labels.append(&mut group.labels);
    }

    Ok(result)
}

fn pack_length_group(
    sequences: &[Vec<u32>],
    max_length: usize,
    pad_token_id: u32,
) -> PyResult<PackedBatch> {
    let mut result = PackedBatch::new();
    let mut current_batch = Vec::new();
    let mut current_length = 0;

    for seq in sequences {
        if current_length + seq.len() <= max_length {
            // Add sequence to current batch
            current_batch.extend(seq);
            current_length += seq.len();
        } else {
            // Pad and add current batch
            add_padded_batch(
                &mut result,
                current_batch,
                max_length,
                pad_token_id,
            )?;
            // Start new batch with current sequence
            current_batch = seq.clone();
            current_length = seq.len();
        }
    }

    // Handle last batch
    if !current_batch.is_empty() {
        add_padded_batch(
            &mut result,
            current_batch,
            max_length,
            pad_token_id,
        )?;
    }

    Ok(result)
}

fn add_padded_batch(
    result: &mut PackedBatch,
    mut batch: Vec<u32>,
    max_length: usize,
    pad_token_id: u32,
) -> PyResult<()> {
    if batch.len() > max_length {
        return Err(PyValueError::new_err(
            format!("Sequence length {} exceeds maximum length {}", batch.len(), max_length)
        ));
    }

    // Create attention mask before padding
    let attention_length = batch.len();
    
    // Pad the batch
    batch.extend(vec![pad_token_id; max_length - batch.len()]);
    
    // Create attention mask and labels
    let mut attention_mask = vec![1u32; attention_length];
    attention_mask.extend(vec![0u32; max_length - attention_length]);
    let labels = vec![-100i32; max_length]; // -100 is typically used for ignored positions

    result.input_ids.push(batch);
    result.attention_mask.push(attention_mask);
    result.labels.push(labels);

    Ok(())
}

#[pyfunction]
pub fn pack_dataset(
    sequences: Vec<Vec<u32>>,
    max_length: usize,
    pad_token_id: u32,
) -> PyResult<(Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<i32>>)> {
    pack_sequences(sequences, max_length, pad_token_id)
        .map(PackedBatch::to_tuple)
}
