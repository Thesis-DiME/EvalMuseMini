echo "START MODEL DOWNLOADING"
./scripts/download_test_dataset.sh
./scripts/download_model.sh
pip install -r requirements.txt
echo "START MODEL DOWNLOADING EVALMUSEMINI"
huggingface-cli download Dnau15/EvalMuseMini --repo-type dataset --local-dir ./data/dataset
python data/preprocessing_v2.py
