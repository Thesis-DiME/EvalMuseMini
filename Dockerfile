FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
 pip install -r requirements.txt

COPY . .

CMD [ "python", "main.py" ]
