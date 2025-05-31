import torch
import json
import hydra
from omegaconf import DictConfig
from pathlib import Path
import csv
from transformers import BertTokenizer
from lavis.models import load_model_and_preprocess
from utils import load_data
from tqdm import tqdm
from typing import List, Dict
from hydra.utils import get_original_cwd


class EvalMusePipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", truncation_side="right"
        )
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.data = load_data(cfg.metadata_path, "json")

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                "fga_blip2", "coco", device=self.device, is_eval=True
            )
        )
        self.model.load_checkpoint(cfg.model_path)
        self.model.eval()

        dataset_dir = Path(cfg.metadata_path).parent
        self.metric = hydra.utils.instantiate(
            cfg.metric,
            model=self.model,
            vis_processors=self.vis_processors,
            text_processors=self.text_processors,
            tokenizer=self.tokenizer,
            dataset_dir=dataset_dir / "images",
        ).to(self.device)

        # Force results path to results/individual_metrics/eval_muse.csv
        self.output_csv_path = Path(
            get_original_cwd(), "results/individual_metrics/eval_muse.csv"
        )
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        for item in tqdm(self.data[:5]):
            self.metric.update(item)

        result_list = self.metric.compute()
        self._save_csv_results(result_list)

    def _save_csv_results(self, result_list: List[Dict]):
        if not result_list:
            return

        # Flatten nested dicts (e.g. score breakdowns)
        flattened_results = []
        for result in result_list:
            flat_result = result.copy()
            for key, value in result.items():
                if isinstance(value, dict):
                    flat_result[key] = json.dumps(value, ensure_ascii=False)
            flattened_results.append(flat_result)

        fieldnames = list(flattened_results[0].keys()) if flattened_results else []

        with open(self.output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_results)
