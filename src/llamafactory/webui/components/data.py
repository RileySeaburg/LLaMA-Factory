# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Generator

from ...extras.constants import DATA_CONFIG
from ...extras.packages import is_gradio_available


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


PAGE_SIZE = 2


def prev_page(page_index: int) -> int:
    return page_index - 1 if page_index > 0 else page_index


def next_page(page_index: int, total_num: int) -> int:
    return page_index + 1 if (page_index + 1) * PAGE_SIZE < total_num else page_index


def can_preview(dataset_dir: str, dataset: list) -> "gr.Button":
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception:
        return gr.Button(interactive=False)

    if len(dataset) == 0 or "file_name" not in dataset_info[dataset[0]]:
        return gr.Button(interactive=False)

    data_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    if os.path.isfile(data_path) or (os.path.isdir(data_path) and os.listdir(data_path)):
        return gr.Button(interactive=True)
    else:
        return gr.Button(interactive=False)


def _count_lines(file_path: str) -> int:
    count = 0
    with open(file_path, 'rb') as f:
        # Count lines efficiently using binary mode
        for _ in f:
            count += 1
    return count


def _lazy_load_json(file_path: str, start: int, size: int) -> List[Any]:
    """Lazily load a specific page from a JSON file"""
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            # For JSONL, we can skip to the desired lines
            for i, line in enumerate(f):
                if i >= start and i < start + size:
                    yield json.loads(line)
                elif i >= start + size:
                    break
        else:
            # For regular JSON, we need to load the array index
            data = json.load(f)
            for item in data[start:start + size]:
                yield item


def _lazy_load_text(file_path: str, start: int, size: int) -> Generator[str, None, None]:
    """Lazily load a specific page from a text file"""
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= start and i < start + size:
                yield line.strip()
            elif i >= start + size:
                break


def get_preview(dataset_dir: str, dataset: list, page_index: int) -> Tuple[int, list, "gr.Column"]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), encoding="utf-8") as f:
        dataset_info = json.load(f)

    data_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    
    # Show loading state
    preview_box = gr.Column(visible=True)
    
    if os.path.isfile(data_path):
        # Get total count efficiently
        total_count = _count_lines(data_path)
        
        # Load just the requested page
        start = PAGE_SIZE * page_index
        if data_path.endswith((".json", ".jsonl")):
            data = list(_lazy_load_json(data_path, start, PAGE_SIZE))
        else:
            data = list(_lazy_load_text(data_path, start, PAGE_SIZE))
    else:
        # Handle directory case
        total_count = 0
        data = []
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            total_count += _count_lines(file_path)
            
            # Only load data if it's in our page range
            start = PAGE_SIZE * page_index
            if len(data) < PAGE_SIZE:
                if file_path.endswith((".json", ".jsonl")):
                    data.extend(list(_lazy_load_json(file_path, start, PAGE_SIZE - len(data))))
                else:
                    data.extend(list(_lazy_load_text(file_path, start, PAGE_SIZE - len(data))))

    return total_count, data, preview_box


def create_preview_box(dataset_dir: "gr.Textbox", dataset: "gr.Dropdown") -> Dict[str, "Component"]:
    data_preview_btn = gr.Button(interactive=False, scale=1)
    with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
        with gr.Row():
            preview_count = gr.Number(value=0, interactive=False, precision=0)
            page_index = gr.Number(value=0, interactive=False, precision=0)

        with gr.Row():
            prev_btn = gr.Button()
            next_btn = gr.Button()
            close_btn = gr.Button()

        with gr.Row():
            preview_samples = gr.JSON()

    dataset.change(can_preview, [dataset_dir, dataset], [data_preview_btn], queue=False).then(
        lambda: 0, outputs=[page_index], queue=False
    )
    data_preview_btn.click(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    prev_btn.click(prev_page, [page_index], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    next_btn.click(next_page, [page_index, preview_count], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    close_btn.click(lambda: gr.Column(visible=False), outputs=[preview_box], queue=False)
    return dict(
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        page_index=page_index,
        prev_btn=prev_btn,
        next_btn=next_btn,
        close_btn=close_btn,
        preview_samples=preview_samples,
    )
