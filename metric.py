import torchmetrics
import os
from PIL import Image
import torch
from utils import get_index
from typing import Dict, Any


class ElementAlignmentMetric(torchmetrics.Metric):
    def __init__(
        self, model, vis_processors, text_processors, tokenizer, dataset_dir, **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.vis_processors = vis_processors
        self.text_processors = text_processors
        self.tokenizer = tokenizer
        self.dataset_dir = dataset_dir
        self.add_state("results", default=[], dist_reduce_fx="cat")

    def update(self, item: Dict[str, Any]):
        # Check if item has the expected structure
        if not isinstance(item, dict):
            raise ValueError(f"Expected item to be a dictionary, got {type(item)}")

        # Handle case where element_score might be a string or improperly formatted

        # if isinstance(element_scores, str):
        #     try:
        #         # Attempt to parse if it's a string representation of a dict
        #         import ast
        #         element_scores = ast.literal_eval(element_scores)
        #     except (ValueError, SyntaxError):
        #         raise ValueError("element_score could not be parsed as a dictionary")

        # if not isinstance(element_scores, dict):
        #     raise ValueError(f"element_score must be a dictionary, got {type(element_scores)}")

        element_scores = item.get("element_score", None)

        prompt = item["prompt"]

        image_path = os.path.join(self.dataset_dir, item["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](image).to(self.device)

        processed_prompt = self.text_processors["eval"](prompt)
        prompt_ids = self.tokenizer(processed_prompt).input_ids

        with torch.no_grad():
            alignment_score, scores = self.model.element_score(
                image.unsqueeze(0), [processed_prompt]
            )

        if element_scores:
            elements = element_scores.keys()
            elements_score = {}
            for element in elements:
                element_ = element.rpartition("(")[0]
                element_ids = self.tokenizer(element_).input_ids[1:-1]

                idx = get_index(element_ids, prompt_ids)

                if idx:
                    mask = [0] * len(prompt_ids)
                    mask[idx : idx + len(element_ids)] = [1] * len(element_ids)
                    mask = torch.tensor(mask, device=self.device)
                    elements_score[element] = (
                        scores * mask
                    ).sum().item() / mask.sum().item()
                else:
                    elements_score[element] = 0

            item["element_result"] = elements_score

        item["score_result"] = alignment_score.item()

        self.results.append(item)

    def compute(self):
        return self.results

    def reset(self):
        self.results = []
