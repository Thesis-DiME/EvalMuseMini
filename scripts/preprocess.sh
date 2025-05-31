./download_test_dataset.sh
./download_model.sh
huggingface-cli download Dnau15/EvalMuseMini --repo-type dataset --local-dir ./data/dataset
python data/preprocessing_v2.py