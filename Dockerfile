FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 wget  -y

COPY requirements.txt  .

RUN --mount=type=cache,target=/root/.cache/pip \
 pip install -r requirements.txt

COPY . .

RUN chmod u+x scripts/preprocess.sh scripts/download.sh scripts/download_model.sh scripts/download_test_dataset.sh && ./scripts/preprocess.sh

CMD [ "python", "main.py" ]
