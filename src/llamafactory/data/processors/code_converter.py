"""Python wrapper for the Rust code converter."""

import json
import os
import sys
from typing import List, Dict, Any

from datasets import Dataset, Features, Value, Sequence

# Add the compiled library directory to Python path
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_lib_dir)))), "target", "release")
if _target_dir not in sys.path:
    sys.path.append(_target_dir)

try:
    from rust_converter import (
        convert_code_data,
        batch_convert_code_data,
        fast_tokenize,
        fast_decode,
        pack_dataset,
    )
except ImportError as e:
    import subprocess
    
    def _build_rust_module():
        """Build the Rust module if not already built."""
        rust_dir = os.path.dirname(_lib_dir)
        try:
            subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True)
            print("Successfully built Rust converter module")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build Rust module: {e}", file=sys.stderr)
            raise ImportError("Failed to build Rust converter module") from e
    
    try:
        _build_rust_module()
        from rust_converter import (
            convert_code_data,
            batch_convert_code_data,
            fast_tokenize,
            fast_decode,
            pack_dataset,
        )
    except (ImportError, subprocess.CalledProcessError) as e:
        def _not_built(*args, **kwargs):
            raise ImportError(
                "Rust converter module not available. Please ensure Rust is installed "
                "and run 'cargo build --release' in the rust_converter directory"
            ) from e
        convert_code_data = batch_convert_code_data = fast_tokenize = fast_decode = pack_dataset = _not_built

def truncate_text(text: str, max_length: int = 32000) -> str:
    """Truncate text to approximately max_length tokens."""
    # Rough estimate: average token is ~4 characters
    char_limit = max_length * 4
    if len(text) > char_limit:
        return text[:char_limit]
    return text

def convert_code_dataset(examples: Dict[str, List[Any]], format_type: str = "alpaca", dataset_type: str = None) -> Dict[str, List[Any]]:
    """Convert code dataset to training format.
    
    Args:
        examples: Dictionary containing dataset examples
        format_type: Target format ("alpaca" or "sharegpt")
        dataset_type: Type of dataset ("starcoder" or "rust_explanation")
        
    Returns:
        Converted dataset in the specified format with fields:
        _prompt, _response, _system, _tools, _images, _videos
    """
    # Prepare input data based on dataset type
    input_data = []
    if dataset_type == "starcoder":
        input_data = [
            json.dumps({"content": truncate_text(content)})
            for content in examples["content"]
        ]
    elif dataset_type == "rust_explanation":
        input_data = [
            json.dumps({
                "input": truncate_text(input_text),
                "output": truncate_text(output_text)
            })
            for input_text, output_text in zip(examples["input"], examples["output"])
        ]
    else:
        # Default handling for code datasets
        input_data = [
            json.dumps({
                "content": truncate_text(content),
                "role": role
            })
            for content, role in zip(examples["content"], examples.get("role", ["user"] * len(examples["content"])))
        ]
    
    # Convert using Rust module
    result = json.loads(batch_convert_code_data(input_data, format_type, dataset_type or ""))
    data = result["data"]
    
    # Create dataset with proper features
    features = Features({
        "_prompt": Sequence({
            "content": Value("string"),
            "role": Value("string")
        }),
        "_response": Sequence({
            "content": Value("string"),
            "role": Value("string")
        }),
        "_system": Value("string"),
        "_tools": Value("string"),
        "_images": Value("null"),
        "_videos": Value("null")
    })
    
    # Ensure proper format for supervised training
    output = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": []
    }
    
    for i in range(len(data.get("_prompt", []))):
        # Ensure _prompt has odd length by adding a user message if needed
        prompt = data["_prompt"][i]
        if len(prompt) % 2 == 0:
            prompt.append({"content": "", "role": "user"})
        output["_prompt"].append(prompt)
        
        # Ensure _response has exactly one message
        response = data["_response"][i]
        if isinstance(response, list) and len(response) > 0:
            output["_response"].append([response[0]])
        else:
            # Default response if none provided
            output["_response"].append([{"content": "", "role": "assistant"}])
        
        # Copy other fields
        output["_system"].append(data.get("_system", [""])[i])
        output["_tools"].append(data.get("_tools", [""])[i])
        output["_images"].append(data.get("_images", [None])[i])
        output["_videos"].append(data.get("_videos", [None])[i])
    
    dataset = Dataset.from_dict(output, features=features)
    return dataset.to_dict()
