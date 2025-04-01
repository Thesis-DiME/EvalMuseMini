import torch
import json
from transformers import BertTokenizer
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import os
from PIL import Image
from utils import load_data, get_index
import argparse
import csv

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})


def eval(args):
    data = load_data(args.data_file, "json")

    model, vis_processors, text_processors = load_model_and_preprocess(
        "fga_blip2", "coco", device=device, is_eval=True
    )
    model.load_checkpoint(args.model_path)
    model.eval()

    result_list = []
    for item in tqdm(data[: args.num_files]):
        elements = item["element_score"].keys()
        prompt = item["prompt"]

        image = os.path.join(args.dataset_dir, item["img_path"])

        image = Image.open(image).convert("RGB")
        image = vis_processors["eval"](image).to(device)
        prompt = text_processors["eval"](prompt)
        prompt_ids = tokenizer(prompt).input_ids

        torch.cuda.empty_cache()
        with torch.no_grad():
            alignment_score, scores = model.element_score(image.unsqueeze(0), [prompt])

        elements_score = dict()
        for element in elements:
            element_ = element.rpartition("(")[0]
            element_ids = tokenizer(element_).input_ids[1:-1]

            idx = get_index(element_ids, prompt_ids)

            if idx:
                mask = [0] * len(prompt_ids)
                mask[idx : idx + len(element_ids)] = [1] * len(element_ids)

                mask = torch.tensor(mask).to(device)
                elements_score[element] = ((scores * mask).sum() / mask.sum()).item()
            else:
                elements_score[element] = 0
        item["score_result"] = alignment_score.item()
        item["element_result"] = elements_score

        result_list.append(item)
    
    with open(args.save_path, "w", newline="", encoding="utf-8") as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)
        
    with open(args.save_path.replace('.json', '.csv'), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result_list[0].keys() if result_list else [])
        
        writer.writeheader()
        
        writer.writerows(result_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/dataset/test.json")
    parser.add_argument("--save_path", type=str, default="results/result.json")
    parser.add_argument(
        "--model_path", type=str, default="pretrained_models/fga_blip2.pth"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="data/dataset/images"
    )
    parser.add_argument("--num_files", type=int, default=3)
    args = parser.parse_args()
    eval(args)
