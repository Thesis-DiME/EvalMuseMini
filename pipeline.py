import torch
import json
from transformers import BertTokenizer
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import os
from PIL import Image
from submodules.EvalMuseMini.utils import load_data, get_index
import hydra
from omegaconf import DictConfig, OmegaConf
import csv
import torchmetrics
from typing import List, Dict
from pathlib import Path


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})


class EvalMusePipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", truncation_side="right"
        )
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.data = load_data(cfg.evaluation.eval_muse.data_file, "json")

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                "fga_blip2", "coco", device=self.device, is_eval=True
            )
        )
        self.model.load_checkpoint(cfg.evaluation.eval_muse.model_path)
        self.model.eval()

        dataset_dir = Path(cfg.metadata_path).parent
        self.metric = hydra.utils.instantiate(
            cfg.evaluation.eval_muse.metric,
            model=self.model,
            vis_processors=self.vis_processors,
            text_processors=self.text_processors,
            tokenizer=self.tokenizer,
            dataset_dir=dataset_dir,
            # _convert_="partial"
        ).to(self.device)

    def evaluate(self):
        for item in tqdm(self.data):
            self.metric.update(item)

        result_list = self.metric.compute()

        self._save_results(result_list)
        self._save_csv_results(result_list)

    def _save_csv_results(self, result_list: List[Dict]):
        if not result_list:
            return

        flattened_results = []
        for result in result_list:
            flat_result = result.copy()
            for key, value in result.items():
                if isinstance(value, dict):
                    flat_result[key] = json.dumps(value, ensure_ascii=False)
            flattened_results.append(flat_result)

        existing_data = []
        file_exists = Path(self.cfg.csv_path).exists()

        if file_exists:
            with open(self.cfg.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

            if len(existing_data) != len(flattened_results):
                raise ValueError(
                    f"Cannot append columns: Existing file has {len(existing_data)} rows "
                    f"but new data has {len(flattened_results)} rows"
                )

            for existing_row, new_row in zip(existing_data, flattened_results):
                existing_row.update(new_row)

            fieldnames = list(existing_data[0].keys()) if existing_data else []
            new_columns = [
                col for col in flattened_results[0].keys() if col not in fieldnames
            ]
            fieldnames.extend(new_columns)
        else:
            existing_data = flattened_results
            fieldnames = list(flattened_results[0].keys()) if flattened_results else []

        with open(self.cfg.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)

    def _save_results(self, result_list):
        with open(self.cfg.save_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    eval()
