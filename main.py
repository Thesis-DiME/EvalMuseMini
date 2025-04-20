import torch
import json
from transformers import BertTokenizer
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from PIL import Image
import hydra
from omegaconf import DictConfig
import csv
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset, DataLoader
from enum import Enum


class ModelName(Enum):
    DeepFloyd_I_XL_v1 = "DeepFloyd_I_XL_v1"
    Midjourney_6 = "Midjourney_6"
    DALLE_3 = "DALLE_3"
    SDXL_Turbo = "SDXL_Turbo"
    SDXL_Base = "SDXL_Base"
    SDXL_2_1 = "SDXL_2_1"


class HumanRatedImageDataset(Dataset):
    def __init__(
        self, metadata_path, folder_path, model_name: ModelName, transform=None
    ):
        self.folder_path = folder_path
        self.model_name = model_name
        self.transform = transform

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self.metadata[idx]

        try:
            # Convert model_name enum to string for dictionary lookup
            relative_img_path = data["Images"][self.model_name.value]
            full_img_path = os.path.join(self.folder_path, relative_img_path)
            image = Image.open(full_img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Error loading image for model {self.model_name.value} at index {idx}: {e}"
            )

        if self.transform:
            image = self.transform(image)

        return {
            "index": data["Index"],
            "prompt": data["Prompt"],
            "image_path": full_img_path,
            "real_image": image,
            "generated_image": image,  # if applicable, or can be removed
            "human_ratings": data["HumanRatings"].get(self.model_name.value, []),
        }
    
class EvalMusePipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        # Load BLIP2 model and processors
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            "fga_blip2", "coco", device=self.device, is_eval=True
        )
        self.model.load_checkpoint(cfg.model_path)
        self.model.eval()

        # Dataset and DataLoader
        dataset_class = hydra.utils.get_class(cfg.dataset._target_)
        model_name = ModelName[cfg.dataset.model_name]
        transform = self.vis_processors["eval"]

        dataset = dataset_class(
            cfg.dataset.metadata_path,
            folder_path=cfg.dataset.folder_path,
            model_name=model_name,
            transform=transform
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )

        # Metric instantiation
        self.metric = hydra.utils.instantiate(
            cfg.metric,
            model=self.model,
            vis_processors=self.vis_processors,
            text_processors=self.text_processors,
            tokenizer=self.tokenizer,
            dataset_dir=cfg.metric.dataset_dir,
        ).to(self.device)

    def evaluate(self):
        for batch in tqdm(self.dataloader):
            prompts = batch["prompt"]
            images = batch["image_path"]

            for prompt, image in zip(prompts, images):
                self.metric.update({
                    "prompt": prompt,
                    "image_path": image,
                })

        result_list = self.metric.compute()
        self._save_results(result_list)
        self._save_csv_results(result_list)

    def _save_csv_results(self, result_list):
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
        file_exists = os.path.exists(self.cfg.csv_path)

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
            new_columns = [col for col in flattened_results[0].keys() if col not in fieldnames]
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


@hydra.main(config_path="conf", config_name="eval_muse")
def main(cfg: DictConfig):
    pipeline = EvalMusePipeline(cfg)
    pipeline.evaluate()


if __name__ == "__main__":
    main()
