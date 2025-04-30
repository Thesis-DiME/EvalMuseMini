from datasets import load_dataset

ds = load_dataset("parquet", data_files="./data/gen_eval/data/train-*.parquet", split="train")

ds.save_to_disk("./data/gen_eval/dataset")
