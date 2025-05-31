scripts/download_test_dataset.sh
scripts/download_model.sh
pip install -r requirements.txt
huggingface-cli download Dnau15/EvalMuseMini --repo-type dataset --local-dir ./data/dataset
python data/preprocessing_v2.py